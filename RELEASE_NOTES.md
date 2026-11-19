# tensorflow-lua 0.18.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- tensor-backed `Dataset.from_tensor(x, y)` with batching, shuffle, and
  `each` iteration;
- no full-dataset Lua-table conversion;
- `Sequential:fit_dataset`, `evaluate_dataset`, and `predict_dataset` route
  tensor datasets directly to their tensor paths;
- new `predict_tensor` method;
- 171 dependency-free tests.

The 0.17 feature set remains: attention/transformer layers, trainable composed
Conv3D, 3-D pooling, functional-model checkpointing, gradient accumulation,
AUC/macro metrics, and native LR schedules.

## Limit

Tensor dataset batching uses byte-level row slicing; tensors must be contiguous
(the normal placeholder/feed path is).

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.18.0-1.rockspec
```
