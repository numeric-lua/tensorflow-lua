# tensorflow-lua 2.0.0

Native LuaJIT bindings for the TensorFlow C API.

## Major addition: portable deployment bundles

A bundle is a self-contained CPU deployment directory:

```text
model.bundle/
  graph.pb
  variables.index
  variables.data-00000-of-00001
  manifest.lua
```

`model:save_bundle(directory)` writes it for Sequential and Functional models.
`tf.load_bundle(directory)` imports the graph, restores the native checkpoint,
and returns a runnable graph/session plus input/output names.

This is **not** official SavedModel export. The TensorFlow C API does not expose
the MetaGraphDef writer. The bundle is the supported deployment artifact for
this Lua runtime.

## Surface

The 1.0 API policy still applies. The binding includes 2-D/3-D
convolution/pooling/upsampling, attention/transformer layers, unrolled
LSTM/GRU, tensor datasets, gradient accumulation, AUC/macro metrics, functional
and Sequential models, and native LR schedules.

## Verification

- 175 dependency-free tests;
- CI on Ubuntu 22.04/24.04 with libtensorflow 2.16.2/2.18.0.

## Limits

Linux x86-64 CPU; no GPU placement; no official SavedModel export; composed
Conv3D is slower than a fused kernel; 2.19/2.20 are unverified.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-2.0.0-1.rockspec
```
