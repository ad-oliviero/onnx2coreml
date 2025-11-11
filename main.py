import os
import shutil
import subprocess as sp

import coremltools as ct
import onnx
from onnxsim import simplify


class MlModel(object):
    def __init__(
        self,
        file,
        author,
        license,
        short_description,
        version,
        input_desc,
        output_desc,
        input_shapes=None,
    ) -> None:
        self.file = file
        self.author = author
        self.license = license
        self.short_description = short_description
        self.version = version
        self.input_desc = input_desc
        self.output_desc = output_desc
        self.input_shapes = input_shapes
        self.simp_file = file.replace(".onnx", "-simplified.onnx")
        self.coreml_file = self.simp_file.replace(".onnx", ".mlpackage")
        self.saved_model_dir = "saved_model"

    def convert(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)
        model = onnx.load(self.file)
        model_simp, check = simplify(
            model,
            overwrite_input_shapes=self.input_shapes,
            dynamic_input_shape=False,
            skip_fuse_bn=True,
            perform_optimization=True,
        )

        if check:
            onnx.save(model_simp, self.simp_file)
            print(f"Simplified {self.file} to {self.simp_file}")
        else:
            raise RuntimeError(f"Failed to simplify {self.file}")

        sp.run(
            f"onnx2tf -osd -nuo -dgc -v info -i {self.simp_file} -o {self.saved_model_dir}",
            shell=True,
            check=True,
        )
        model = ct.convert(
            self.saved_model_dir,
            source="tensorflow",
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
        )

        spec = model.get_spec()

        model.author = self.author
        model.license = self.license
        model.short_description = self.short_description
        model.version = self.version

        output_name = spec.description.output[0].name
        input_name = spec.description.input[0].name

        model.input_description[input_name] = self.output_desc
        model.output_description[output_name] = self.output_desc

        self.coreml_file = self.coreml_file.replace("-simplified", "")
        model.save(self.coreml_file)
        print(f"Saved {self.file} to {self.coreml_file}")

        shutil.rmtree(self.saved_model_dir)


if __name__ == "__main__":
    # Right now this is how you set the models to convert,
    # in future (if I keep to maintain this project) it will be easier
    src_models = [
        MlModel(
            "pix2text-mfd-1.5/pix2text-mfd-1.5.onnx",
            "breezedeus",
            "MIT",
            "Mathematical Formula Detection (MFD) model from Pix2Text (P2T)",
            "1.5",
            "Input image (768x768 RGB), values normalized to [0, 1]",
            "Detection output (1, 6, 12096) - bounding boxes and scores",
            input_shapes={"images": [1, 3, 768, 768]},
        ),
        MlModel(
            "pix2text-mfr-1.5/decoder_model.onnx",
            "breezedeus",
            "MIT",
            "Mathematical Formula Recognition (MFR) model from Pix2Text (P2T) - Decoder",
            "1.5",
            "Encoder hidden states and input_ids",
            "Logits for next token",
            input_shapes={
                "input_ids": [1, 1],
                "encoder_hidden_states": [1, 576, 384],
            },
        ),
        MlModel(
            "pix2text-mfr-1.5/encoder_model.onnx",
            "breezedeus",
            "MIT",
            "Mathematical Formula Recognition (MFR) model from Pix2Text (P2T) - Encoder",
            "1.5",
            "Input image (384x384 RGB), values normalized",
            "Encoder hidden states",
            input_shapes={"pixel_values": [1, 3, 384, 384]},
        ),
    ]
    for model in src_models:
        model.convert()
