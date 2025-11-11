import subprocess as sp

import coremltools as ct
import onnx
from onnxsim import simplify


def convert(model_name: [str, str, str]):
    model_name[1] = model_name[0].replace(".onnx", "-simplified.onnx")
    model = onnx.load(model_name[0])

    model_simp, check = simplify(
        model,
        input_shapes={"images": [1, 3, 768, 768]},
        dynamic_input_shape=False,
        skip_fuse_bn=True,
    )

    if check:
        onnx.save(model_simp, model_name[1])
        print(f"Saved {model_name[0]} to {model_name[1]}")
    else:
        raise

    sp.run(f"onnx2tf -i {model_name[1]}", shell=True, check=True)
    model_name[2] = model_name[0].replace(".onnx", ".mlpackage")
    model = ct.convert(
        "saved_model",
        source="tensorflow",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    spec = model.get_spec()
    print("\nInputs:")
    for input_desc in spec.description.input:
        print(f"  Name: {input_desc.name}")
        print(f"  Type: {input_desc.type}")

    print("\nOutputs:")
    for output_desc in spec.description.output:
        print(f"  Name: {output_desc.name}")
        print(f"  Type: {output_desc.type}")

    model.author = "Pix2Text-MFD"
    model.license = "Apache 2.0"
    model.short_description = "Mathematical Formula Detection Model"
    model.version = "1.5"

    output_name = spec.description.output[0].name
    input_name = spec.description.input[0].name

    print(f"\nActual input name: {input_name}")
    print(f"Actual output name: {output_name}")

    model.input_description[input_name] = (
        "Input image (768x768 RGB), values normalized to [0, 1]"
    )
    model.output_description[output_name] = (
        "Detection output (1, 6, 12096) - bounding boxes and scores"
    )

    model.save(model_name[2])

    print(f"Saved as: {model_name[2]}")
    sp.run("rm -r saved_model", shell=True, check=True)


if __name__ == "__main__":
    # Right now this is how you set the models to convert,
    # in future (if I keep to maintain this project) it will be easier
    model_names = [
        ["pix2text-mfd-1.5/pix2text-mfd-1.5.onnx", "", ""],
        # ["", "", ""],
        # ["", "", ""],
    ]
    for nm in model_names:
        convert(nm)
