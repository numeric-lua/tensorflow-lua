# tensorflow-lua 0.9.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- Sequential architecture configuration:
  - `get_config()`, `save_config(path)`, `load_config(path)`,
    `Sequential.from_config(table)`;
  - Lua-literal configuration format with cycle/function rejection;
  - architecture and compile options survive without retyping the model;
  - weights remain separate and can be restored with existing checkpoints.

The 0.8 feature set remains:

- Reshape, Permute, UpSampling2D, ZeroPadding2D, DepthwiseConv2D,
  GlobalMaxPool1D, Conv1D/Conv2D, pooling, BatchNorm, Activation,
  Embedding, Flatten, and Dropout;
- global-norm gradient clipping;
- feedable learning rates and ReduceLROnPlateau;
- top-k metrics, sample weights, label smoothing, training-state checkpoints;
- libtensorflow 2.16.2/2.18.0 compatibility.

## Requirements

- LuaJIT 2.1
- libtensorflow 2.16.2 or 2.18.0 CPU for Linux x86-64
- Lua 5.1-compatible LuaRocks

## Install

```sh
luarocks --lua-version=5.1 install tensorflow-lua
```

## Verification

- 141 dependency-free tests;
- luacheck clean;
- CI Ubuntu 22.04/24.04 × libtensorflow 2.16.2/2.18.0.
