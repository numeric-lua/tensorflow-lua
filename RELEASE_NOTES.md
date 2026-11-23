# tensorflow-lua 2.3.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Selected GraphDef outputs.** `graph:import_graph_def(bytes, { prefix =
  "left", return_elements = { "left/z:0" } })` returns graph outputs directly,
  with operation, index, name, dtype, and inferred shape, instead of only the
  graph. `import_file` accepts the same option.
- **Deeper graph introspection.** `graph:consumers(output)` lists every
  consumer and its input index. `graph:control_inputs(operation)` and
  `graph:control_outputs(operation)` expose control edges, and
  `graph:operation_info` now includes `consumer_ops`, `control_inputs`, and
  `control_outputs`.
- **Broader eager execution.** Eager tensors now support `concat`, `gather`,
  `slice`, `pad` (including `REFLECT`/`SYMMETRIC` mirror padding), `tile`,
  `where`, `reverse`, and `split`. The generic `execute` path supports string
  attributes (`{ string = "CONSTANT" }`) and multi-output ops through
  `{ num_outputs = N }`.
- **AdaMax optimizer.** `tf.train.adamax` adds `ApplyAdaMax` updates with
  first-moment, infinity-norm, and beta1-power state, optional gradient-value
  clipping, and initialization metadata.
- **More typed graph ops.** `tf.stack`, `tf.unstack`, `tf.slice`, and
  `tf.reverse` cover packing, unpacking, slicing, and axis reversal.

## Surface

The 1.0 API policy still applies; 2.3 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

## Verification

- 203 dependency-free tests;
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
luarocks --lua-version=5.1 make tensorflow-lua-2.3.0-1.rockspec
```
