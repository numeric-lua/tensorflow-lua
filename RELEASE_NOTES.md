# tensorflow-lua 1.0.0

Native LuaJIT bindings for the TensorFlow C API.

## Milestone

The Lua-facing API is now documented as stable and follows semantic versioning.
See `docs/stability.md`.

## Included surface

- tensors, graphs, sessions, SavedModel loading;
- graph construction, variables, gradients, optimizers, checkpoints;
- 2-D/3-D convolution, pooling, upsampling, batch/layer normalization;
- attention, multi-head attention, transformer blocks, positional encoding;
- unrolled LSTM/GRU;
- Sequential and functional models;
- tensor-backed datasets, callbacks, metrics, losses, schedules;
- gradient accumulation, AUC, macro multiclass PRF.

## Test status

- 174 dependency-free tests;
- CI on Ubuntu 22.04/24.04 with libtensorflow 2.16.2/2.18.0.

## Platform limits

Linux x86-64 CPU; no GPU placement; no official SavedModel export; Conv3D is
composed and slower than a fused native kernel; 2.19/2.20 are not verified.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-1.0.0-1.rockspec
```
