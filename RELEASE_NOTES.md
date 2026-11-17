# tensorflow-lua 0.8.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- Reshape, Permute, UpSampling2D, ZeroPadding2D, DepthwiseConv2D, and
  GlobalMaxPool1D layers;
- depthwise convolution stacks in Sequential;
- global-norm gradient clipping available to every optimizer;
- shape and parameter tracking for the new layers;
- 140 dependency-free tests.

The 0.7 feature set remains:

- feedable learning rates and `ReduceLROnPlateau`;
- training-state `ModelCheckpoint`;
- Conv1D/Conv2D, pooling, BatchNormalization, Activation, Embedding, Flatten,
  and Dropout layers;
- top-k accuracy metrics;
- dense and sparse sample weights, class weights, and sparse label smoothing;
- resumable optimizer and BatchNorm state;
- libtensorflow 2.16.2/2.18.0 compatibility.

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
`luarocks make tensorflow-lua-0.8.0-1.rockspec` from a checkout.

## Verification

- 140 dependency-free tests
- luacheck clean
- CI: Ubuntu 22.04/24.04, libtensorflow 2.16.2/2.18.0, LuaRocks load
