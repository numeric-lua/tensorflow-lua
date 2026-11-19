# tensorflow-lua 0.16.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- scaled dot-product attention over rank-3 `[batch, time, features]` inputs;
- multi-head attention with query/key/value projections, per-head weights,
  output projection, and optional causal masking;
- attention is composed from portable TensorFlow ops and is trainable;
- 169 dependency-free tests.

The 0.15 feature set remains: trainable composed Conv3D, 3-D pooling,
functional-model checkpointing, gradient accumulation, AUC/macro metrics,
tensor mini-batches, separable convolution, and native LR schedules.

## API notes

`tf.layers.attention` requires `query_shape` and `key_shape` as
`{ time, features }`. `tf.layers.multi_head_attention` additionally takes
`num_heads`, `key_dim`, `value_dim`, and `output_dim`.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.16.0-1.rockspec
```
