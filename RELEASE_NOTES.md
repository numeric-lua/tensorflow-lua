# tensorflow-lua 0.14.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- Conv3D, MaxPool3D, AvgPool3D, GlobalAvgPool3D, and GlobalMaxPool3D layers;
- NDHWC shape tracking and summaries in Sequential;
- `compile({ optimizer = "none" })` for inference-only models;
- 165 dependency-free tests.

## Important limit

Conv3D is **inference-only**: the TensorFlow C library used here does not
register a gradient for the `Conv3D` op. Building a trainable Conv3D model with
an optimizer fails in `tf.gradients`; use inference-only compilation for 3-D
prediction. MaxPool3D/AvgPool3D and the global 3-D pools are covered by the
layer tests.

The 0.13/0.12 feature set remains: functional model checkpointing, gradient
accumulation, AUC/macro metrics, tensor mini-batches, separable convolution,
layer normalization, and native LR schedules.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.14.0-1.rockspec
```
