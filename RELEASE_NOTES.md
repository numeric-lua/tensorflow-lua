# tensorflow-lua 0.10.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- `Sequential:fit_tensor(x, y, opts)` and `evaluate_tensor(x, y)` for
  full-batch Tensor feeds;
- avoids Lua-table conversion for large inputs;
- retains callbacks, BatchNorm updates, metric history, and evaluation.

This is a full-batch path, not a mini-batch iterator. Table-based `fit` remains
available for shuffled mini-batches.

The 0.9 feature set remains:

- architecture config save/load;
- Reshape, Permute, UpSampling2D, ZeroPadding2D, DepthwiseConv2D,
  GlobalMaxPool1D, Conv1D/Conv2D, pooling, BatchNorm, Activation, Embedding,
  Flatten, and Dropout;
- global-norm gradient clipping and model checkpoints;
- libtensorflow 2.16.2/2.18.0 compatibility.

## Requirements

- LuaJIT 2.1
- libtensorflow 2.16.2 or 2.18.0 CPU for Linux x86-64

## Install

```sh
luarocks --lua-version=5.1 install tensorflow-lua
```

## Verification

- 142 dependency-free tests;
- luacheck clean;
- CI Ubuntu 22.04/24.04 × libtensorflow 2.16.2/2.18.0.
