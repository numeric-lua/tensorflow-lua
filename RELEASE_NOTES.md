# tensorflow-lua 0.11.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- tensor-backed mini-batch `fit_tensor`/`evaluate_tensor` (no Lua table
  conversion) with slicing, shuffle, and seed;
- `tf.Functional` multi-input/multi-output graph models with custom losses and
  optimizers;
- MaxPool1D, AvgPool1D, GlobalAvgPool1D, SeparableConv2D,
  LayerNormalization, and Embedding padding rows;
- Huber, hinge, and categorical-hinge losses, dense class weights;
- precision, recall, and F1 metrics;
- exponential, cosine, polynomial, piecewise, and constant LR schedules;
- 157 dependency-free tests.

## Deliberately not included

- SavedModel export (the TensorFlow C API lacks the MetaGraphDef writer);
- GPU placement on this machine;
- graph-level gradient accumulation;
- AUC and full multi-class precision/recall/F1;
- verified libtensorflow 2.19/2.20 compatibility.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.11.0-1.rockspec
```

The binding requires LuaJIT 2.1 and libtensorflow 2.16.2 or 2.18.0 on Linux
x86-64. Set `TENSORFLOW_ROOT` or `TENSORFLOW_LIB` when the library is not in
`third_party/libtensorflow`.
