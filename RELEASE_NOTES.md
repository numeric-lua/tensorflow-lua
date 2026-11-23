# tensorflow-lua 2.2.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Remappable GraphDef imports.** `graph:import_graph_def(bytes, { input_map =
  { x = replacement } })` and `import_file` connect imported placeholders to
  existing graph outputs, so a serialized graph can be composed in Lua without
  feeding every imported input.
- **Graph introspection.** `graph:operations()`, `graph:shape(output)`, and
  `graph:operation_info(output)` expose operation types, inputs, outputs,
  dtypes, consumer counts, and inferred shapes. Invalid output indices raise a
  Lua error before reaching TensorFlow.
- **Broader eager execution.** Eager tensors now support
  `reduce_sum`/`reduce_mean`/`reduce_max`/`reduce_min`/`reduce_prod` with
  `keep_dims`, plus `exp`, `log`, `sqrt`, `abs`, `tanh`, `softplus`, `softmax`,
  `elu`, and `leaky_relu`.
- **Sequence and selection ops.** `tf.split` uses `Split` for equal partitions
  and `SplitV` for explicit sizes; `tf.pad` supports `CONSTANT`, `REFLECT`, and
  `SYMMETRIC`; `tf.tile` and `tf.where` add tiling and broadcasting selection.
- **Custom gradient seeds.** `tf.gradients(graph, outputs, inputs, grad_ys)`
  accepts one upstream gradient per output and correctly combines gradients
  from multiple outputs.

## Surface

The 1.0 API policy still applies; 2.2 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

## Verification

- 196 dependency-free tests;
- CI on Ubuntu 22.04/24.04 with libtensorflow 2.16.2/2.18.0/2.18.1.

## Limits

Linux x86-64 CPU; no GPU placement; no official SavedModel export; composed
Conv3D is slower than a fused kernel. libtensorflow 2.19/2.20 are unsupported
because Google stopped publishing Linux x86-64 C archives after 2.18.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-2.2.0-1.rockspec
```
