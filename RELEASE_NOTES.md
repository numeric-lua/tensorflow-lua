# tensorflow-lua 2.5.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- **Attribute-name introspection.** `graph:attribute_names(output)` enumerates
  the attributes attached to any operation, complementing the existing
  `attribute_info`/`attribute` value readers.
- **ArgMin and Relu6.** Graph and eager surfaces add `argmin` and `relu6`,
  with matching axis validation for `argmax`/`argmin`.
- **List addition.** `tf.add_n` and `Context:add_n` sum non-empty graph-output
  or eager-tensor lists.
- **Stable log-sum-exp.** `tf.reduce_logsumexp` and
  `Context:reduce_logsumexp` use a max-shifted implementation, support
  `keep_dims`, and avoid overflow for large inputs.
- **Nesterov momentum and clipping.** `tf.train.momentum` accepts
  `nesterov = true`/`use_nesterov = true` and `clipvalue`/`clip_value` for
  per-gradient value clipping, in addition to the existing `clipnorm` path.

## Surface

The 1.0 API policy still applies; 2.5 is a minor release with additions only.
The binding includes 2-D/3-D convolution/pooling/upsampling,
attention/transformer layers, unrolled LSTM/GRU, tensor datasets, gradient
accumulation, AUC/macro metrics, functional and Sequential models, native LR
schedules, and portable deployment bundles.

`tf.reduce_logsumexp` is composed from portable TensorFlow ops
(`Max`/`Sub`/`Exp`/`Sum`/`Log`/`AddV2`) and is therefore differentiable
through the normal graph gradient path. It is not a single fused kernel.

## Verification

- 214 dependency-free tests;
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
luarocks --lua-version=5.1 make tensorflow-lua-2.5.0-1.rockspec
```
