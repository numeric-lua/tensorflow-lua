# tensorflow-lua 2.4.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Operation attribute introspection.** `graph:attribute_info(output, name)`
  reports the attribute kind, scalar/list shape, and size;
  `graph:attribute(output, name)` decodes string, int, float, bool, dtype,
  shape, and list attributes directly from a `TF_Operation`.
- **Comparisons and logic.** Graph and eager surfaces now provide
  `equal`/`not_equal`/`greater`/`greater_equal`/`less`/`less_equal`,
  `logical_and`/`logical_or`/`logical_xor`/`logical_not`,
  `squared_difference`, `maximum`, and `minimum`, with `bool` outputs.
- **Extended reductions.** Graph and eager execution add `reduce_prod`,
  `reduce_all`, and `reduce_any`, including `keep_dims` support.
- **Eager structural ops.** `Context:stack`, `Context:unstack`,
  `Context:one_hot`, and `Context:clip_by_value` join the eager convenience
  methods, with the same validation as their graph-mode counterparts.
- **Reusable gradient clipping.** `tf.train.clip_by_norm` and
  `tf.train.clip_by_global_norm` return clipped graph outputs plus the
  unclipped norm; optimizer `clipnorm` now uses the shared global-norm path.

## Surface

The 1.0 API policy still applies; 2.4 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

`tf.logical_xor` is implemented with `NotEqual` because the `LogicalXor` op is
not registered in the supported libtensorflow archives. Tensor, placeholder,
and function attributes are reported by `attribute_info` but are not decoded by
`attribute`.

## Verification

- 209 dependency-free tests;
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
luarocks --lua-version=5.1 make tensorflow-lua-2.4.0-1.rockspec
```
