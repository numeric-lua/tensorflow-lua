# tensorflow-lua 2.6.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Tensor shape refinement.** `graph:set_shape(output, shape)` merges known
  dimensions into an operation's inferred shape, using `-1` for unknown
  dimensions and reporting incompatible shapes through the normal
  `TF_Status` error path.
- **Batched linear algebra.** `tf.batch_matmul` and
  `Context:batch_matmul` wrap `BatchMatMulV2` with `adj_x`/`adj_y` (and
  `transpose_a`/`transpose_b`) options for rank-2 and batched rank-3+ inputs.
- **Log-softmax.** `tf.nn.log_softmax` and `Context:log_softmax` add the
  numerically stable log-softmax activation and preserve float64 inputs.
- **Tensor builders.** `tf.fill`, `tf.range`, and `tf.linspace` join the graph
  surface, with eager equivalents on `tf.eager.Context`.
- **Adam Nesterov momentum.** `tf.train.adam` accepts
  `nesterov = true`/`use_nesterov = true`; `clipvalue` is now validated
  consistently in Adam and its AdamW wrapper.

## Surface

The 1.0 API policy still applies; 2.6 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

`tf.batch_matmul` requires TensorFlow's `BatchMatMulV2` kernel. `tf.fill`,
`tf.range`, and `tf.linspace` are built from the corresponding portable
TensorFlow ops and therefore follow TensorFlow's dtype and shape rules.

## Verification

- 219 dependency-free tests;
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
luarocks --lua-version=5.1 make tensorflow-lua-2.6.0-1.rockspec
```
