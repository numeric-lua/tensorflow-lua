# tensorflow-lua 2.7.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Decoded attribute tables.** `graph:attributes(output)` combines
  `attribute_names`, `attribute_info`, and `attribute` into one table per
  attribute, skipping only the undecoded tensor/placeholder/function kinds.
- **Rounding and reciprocal math.** `tf.floor`, `tf.ceil`, `tf.round`,
  `tf.sign`, `tf.rsqrt`, and `tf.reciprocal` join the graph surface, with eager
  equivalents on `tf.eager.Context`.
- **Variance and standard deviation.** `tf.reduce_variance` and
  `tf.reduce_stddev` support integer axis tables and `keep_dims`; the eager
  context exposes the same reductions.
- **Consistent gradient clipping.** `clipvalue`/`clip_value` now applies to
  RMSProp, CenteredRMSProp, Adagrad, Adadelta, ProximalAdagrad, and Ftrl,
  matching Adam, Adamax, momentum, and gradient descent.

## Surface

The 1.0 API policy still applies; 2.7 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

`tf.reduce_variance` and `tf.reduce_stddev` are composed from portable
TensorFlow ops (`Mean`/`SquaredDifference`/`Sqrt`) and are differentiable
through the normal graph gradient path. The adaptive optimizers clip the
gradient graph outputs before applying their update ops.

## Verification

- 224 dependency-free tests;
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
luarocks --lua-version=5.1 make tensorflow-lua-2.7.0-1.rockspec
```
