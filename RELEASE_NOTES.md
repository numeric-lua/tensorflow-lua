# tensorflow-lua 2.8.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Operation list lengths.** `graph:input_list_length(output, name)` and
  `graph:output_list_length(output, name)` report the arity of list-valued
  operation arguments such as `Pack` inputs and `Split` outputs.
- **Zero/one and gradient control.** `tf.zeros_like`, `tf.ones_like`,
  `tf.identity`, and `tf.stop_gradient` join the graph surface, with eager
  equivalents on `tf.eager.Context`.
- **Error functions and predicates.** `tf.erf`, `tf.erfc`, `tf.is_nan`,
  `tf.is_inf`, and `tf.is_finite` add numeric error functions and boolean
  predicates in graph and eager modes.
- **Checkpoint metadata.** `CheckpointReader:dtype(name)`, `shape(name)`, and
  `info(name)` expose variable metadata without loading the tensor bytes.

## Surface

The 1.0 API policy still applies; 2.8 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

`tf.stop_gradient` maps directly to TensorFlow's `StopGradient` op. The
predicate ops return `bool` tensors, while `erf`/`erfc` preserve their input
floating dtype. Checkpoint metadata comes from
`TF_CheckpointReaderGetVariableDataType` and
`TF_CheckpointReaderGetVariableShape`; missing variables raise Lua errors.

## Verification

- 229 dependency-free tests;
- full suite run with the checkout's libtensorflow 2.18.0;
- CI matrix remains Ubuntu 22.04/24.04 with libtensorflow 2.16.2/2.18.0/2.18.1.

## Limits

Linux x86-64 CPU; no GPU placement; no official SavedModel export; composed
Conv3D is slower than a fused kernel. libtensorflow 2.19/2.20 are unsupported
because Google stopped publishing Linux x86-64 C archives after 2.18.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-2.8.0-1.rockspec
```
