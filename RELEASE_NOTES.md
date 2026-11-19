# tensorflow-lua 0.12.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- graph-level gradient accumulation via
  `compile({ gradient_accumulation_steps = N })` for the supported optimizers;
- binary ROC AUC;
- macro multiclass precision, recall, and F1 over sparse or one-hot labels;
- AUC and macro metrics wired into Sequential `compile({ metrics = ... })`;
- 162 dependency-free tests.

The 0.11 feature set remains: tensor-backed mini-batches, `tf.Functional`
multi-input/multi-output graphs, 1-D pooling, separable convolution, layer
normalization, new losses, and native LR schedules.

## Known limits

- AUC uses an O(N^2) pairwise graph; keep evaluation batches moderate.
- Macro PRF covers integer classes, not per-class reporting tables in
  Sequential history.
- libtensorflow 2.19/2.20 archives were not available at the standard URLs
  during this work, so 2.16.2 and 2.18.0 remain the tested matrix.
- SavedModel export and GPU placement remain out of scope.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.12.0-1.rockspec
```
