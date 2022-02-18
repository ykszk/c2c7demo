# C2C7 Angle Demo
Demo apps of C2C7 Cobb angle measure by AI.

Following two versions are available.

- [Executable]()
- [Web app](https://yk-szk.github.io/c2c7demo/)

# Development

## Executable
Executable varsion is coded in rust. Some functions are exported as wasm for the web app version.


## Create ONNX model for the web app
The model needs to be compatible with WebGL backend.

1. Use `torch.onnx.export` to export to ONNX model
2. Apply `resize_align_corners.py` to change resize mode to align_corners. `pytorch_half_pixel` is not available in WebGL?
3. Apply [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) to optimize away nodes unavailable in WebGL. e.g. `python -m onnxsim model.onnx simplefied.onnx --input-shape 2,1,768,768 --dynamic-input-shape`

(Note to myself) Do not add the model to the repository. Keep the repo lean.