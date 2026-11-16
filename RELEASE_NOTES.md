# tensorflow-lua 0.5.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- Conv2D, MaxPool2D, and BatchNormalization layers, including
  Conv2D/MaxPool/BatchNorm stacks in Sequential;
- runtime train/eval BatchNorm and Dropout switching;
- per-sample weights and label smoothing for sparse classification losses;
- resumable optimizer and BatchNorm moving-statistics checkpoints;
- tested against libtensorflow 2.16.2 and 2.18.0;
- CI matrix across Ubuntu 22.04/24.04 and both TensorFlow versions.

Existing 0.4 features remain:

- GraphDef/SavedModel execution and graph construction;
- Adam/AdamW, RMSProp, CenteredRMSProp, Adagrad, Adadelta, Ftrl,
  ProximalAdagrad, momentum, and gradient descent;
- TensorFlow V2 checkpoint save/restore;
- Dense, Embedding, Flatten, and Dropout layers;
- Sequential fit/evaluate/predict, callbacks, metrics, validation, and
  weight snapshots;
- dense, string, complex, half, bfloat16, and quantized tensors.

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
`luarocks make tensorflow-lua-0.5.0-1.rockspec` from a checkout.

## Verification

- 122 dependency-free tests
- luacheck clean
- CI: Ubuntu 22.04/24.04, libtensorflow 2.16.2/2.18.0, LuaRocks load
