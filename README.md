# onnx2coreml
Onnx (to tensorflow) to Coreml

Since [onnx-coreml](https://github.com/onnx/onnx-coreml) is deprecated, I tried
(and kind of succeeded) to reimplement it in another way:

1. This converts an onnx ml model to a tflite (tensorflow) model
2. Then it converts the tflite model into a mlpackage (coreml) model
