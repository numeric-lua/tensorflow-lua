# tensorflow-lua 0.15.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- trainable Conv3D implemented as a depth-slice composition of portable
  Conv2D kernels; the C library's native Conv3D op has no registered gradient;
- SAME/VALID padding, arbitrary 3-D strides, and dilation;
- NDHWC shape tracking and summaries;
- 166 dependency-free tests.

MaxPool3D, AvgPool3D, GlobalAvgPool3D, and GlobalMaxPool3D remain available.
The 0.14/0.13 feature set is unchanged.

## Important note

The composed Conv3D is correct and trainable but issues one Conv2D per kernel
depth slice, so it is slower than a fused native 3-D kernel. This tradeoff is
documented rather than hidden.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.15.0-1.rockspec
```
