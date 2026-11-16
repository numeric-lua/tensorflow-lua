# tensorflow-lua 0.7.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- feedable optimizer learning rates;
- `ReduceLROnPlateau` callback with patience, factor, cooldown, and minimum
  learning rate;
- `ModelCheckpoint` support for full training state, including optimizer
  slots, BatchNorm statistics, epoch, and learning rate;
- learning rate persisted in resumable training checkpoints;
- release and patch-release checklist.

The 0.6 feature set remains:

- Conv1D/Conv2D, pooling, BatchNormalization, Activation, Embedding, Flatten,
  and Dropout layers;
- Conv1D/Conv2D/pool/BatchNorm stacks in Sequential;
- top-k accuracy metrics;
- per-sample weights for dense and sparse losses;
- label smoothing and class weights for sparse losses;
- runtime train/eval BatchNorm and Dropout;
- libtensorflow 2.16.2/2.18.0 compatibility;
- GraphDef/SavedModel execution, optimizers, checkpoints, and broad tensor
  dtype support.

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
`luarocks make tensorflow-lua-0.7.0-1.rockspec` from a checkout.

## Verification

- 134 dependency-free tests
- luacheck clean
- CI: Ubuntu 22.04/24.04, libtensorflow 2.16.2/2.18.0, LuaRocks load
