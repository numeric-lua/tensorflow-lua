# tensorflow-lua 2.1.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Namespaced GraphDef imports.** `graph:import_graph_def(bytes, { prefix =
  "left" })` and `import_file(path, opts)` place imported operations in a
  namespace, so one graph can hold multiple copies of a serialized model.
- **Explicit eager ownership.** `ETensor:destroy()` releases a handle
  immediately and is idempotent. `Context:close()` defers context deletion until
  every live tensor has drained; unclosed contexts and handles are finalized by
  `ffi.gc` and covered by release counters.
- **Broader eager ops.** `div`, `cast`, `transpose`, `expand_dims`, `squeeze`,
  `reduce_sum`, and `argmax`, plus integer-list op attributes.
- **SGD clipping options.** `tf.train.gradient_descent` now accepts an options
  table with `clipnorm`, `clipvalue`, and precomputed `gradients`.
- **Graph rank/gather ops.** `tf.transpose`, `tf.expand_dims`, `tf.squeeze`, and
  `tf.gather` build the corresponding TensorFlow operations.
- **Packaging fix.** The rockspec includes `tensorflow.bundle`, so a LuaRocks
  install loads the public module successfully.

## Surface

The 1.0 API policy still applies; 2.1 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

## Verification

- 187 dependency-free tests;
- CI on Ubuntu 22.04/24.04 with libtensorflow 2.16.2/2.18.0/2.18.1.

## Limits

Linux x86-64 CPU; no GPU placement; no official SavedModel export; composed
Conv3D is slower than a fused kernel. libtensorflow 2.19/2.20 are unsupported
because Google stopped publishing Linux x86-64 C archives after 2.18.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-2.1.0-1.rockspec
```
