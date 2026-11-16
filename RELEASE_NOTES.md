# tensorflow-lua 0.4.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- GraphDef import/export and SavedModel signature execution.
- A native graph builder with typed attributes, gradients, variables, and
  initializers.
- Optimizers: gradient descent, Adam, AdamW, RMSProp, CenteredRMSProp,
  Adagrad, Adadelta, Ftrl, ProximalAdagrad, and momentum.
- Dense, Embedding, Flatten, and runtime train/eval Dropout layers.
- Sequential `fit`/`evaluate`/`predict`, validation, class weights, metrics,
  callbacks, weight snapshots, and resumable optimizer/epoch state.
- TensorFlow V2 checkpoint save/restore plus official checkpoint reading.
- Dense, string, complex, half, bfloat16, and quantized tensors.
- Experimental eager execution.
- Dependency-free regression suite and CI on Ubuntu 22.04/24.04.

## Requirements

- LuaJIT 2.1
- libtensorflow 2.18 CPU for Linux x86-64 (not bundled)
- Lua 5.1-compatible LuaRocks for optional packaging

## Install

```sh
./scripts/fetch_libtensorflow.sh
# optional:
luarocks --lua-version=5.1 make tensorflow-lua-0.4.0-1.rockspec
```

The 0.4.0 archive checksum is pinned by the bootstrap script.

## Verification

- 113 dependency-free tests
- luacheck clean
- CI: Ubuntu 22.04, Ubuntu 24.04, and LuaRocks package load
