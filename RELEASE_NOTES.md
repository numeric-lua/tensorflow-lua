# tensorflow-lua 0.6.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- Conv1D and rank-3 BatchNormalization;
- average, global-average, and global-max pooling;
- Activation layer;
- top-k accuracy metrics;
- per-sample weights for dense MSE/MAE/BCE/CCE losses;
- streaming native tensor construction without an intermediate flat table;
- documented 0.6 performance baseline.

The 0.5 feature set remains:

- Conv2D/MaxPool2D/BatchNorm stacks and runtime train/eval BatchNorm/Dropout;
- sample weights and label smoothing for sparse losses;
- resumable optimizer and BatchNorm moving-statistics checkpoints;
- libtensorflow 2.16.2/2.18.0 compatibility and Ubuntu 22.04/24.04 CI;
- GraphDef/SavedModel execution, optimizers, checkpoint save/restore, dense,
  string, complex, half, bfloat16, and quantized tensors.

## Requirements

- LuaJIT 2.1
- libtensorflow 2.16.2 or 2.18.0 CPU for Linux x86-64 (not bundled)
- Lua 5.1-compatible LuaRocks for optional packaging

## Install

```sh
./scripts/fetch_libtensorflow.sh
luarocks --lua-version=5.1 install tensorflow-lua
```

The package is published at
<https://luarocks.org/modules/mateus-m-reis/tensorflow-lua>. It can also be
installed from the release source rock or built with
`luarocks make tensorflow-lua-0.6.0-1.rockspec` from a checkout.

## Verification

- 131 dependency-free tests
- luacheck clean
- CI: Ubuntu 22.04/24.04, libtensorflow 2.16.2/2.18.0, LuaRocks load
