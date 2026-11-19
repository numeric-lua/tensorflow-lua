# tensorflow-lua 0.13.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- `tf.Functional` now supports native checkpoint save/load, weight get/set,
  GraphDef export, summaries, callbacks, validation loss, and per-output loss
  weights;
- 163 dependency-free tests.

The 0.12 feature set remains: graph-level gradient accumulation, binary ROC
AUC, macro multiclass PRF, tensor-backed mini-batches, multi-input/multi-output
functional graphs, 1-D/separable convolution, layer normalization, and native
LR schedules.

## Known limits

- `tf.Functional` remains a graph-building API, not a Keras layer registry or
  architecture serializer.
- AUC is O(N^2) in evaluation batch size.
- libtensorflow 2.19/2.20 archives were unavailable (404) at the standard URLs;
  2.16.2 and 2.18.0 remain the tested matrix.
- SavedModel export and GPU placement remain out of scope.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.13.0-1.rockspec
```
