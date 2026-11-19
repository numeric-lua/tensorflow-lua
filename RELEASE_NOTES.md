# tensorflow-lua 0.17.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- sinusoidal positional encoding for `[batch, time, features]` inputs;
- pre-norm transformer block combining multi-head attention, residual
  connections, layer normalization, and a two-layer feed-forward network;
- transformer block is trainable and returns its variables/initializers for
  use with `tf.Functional` or a custom graph;
- 170 dependency-free tests.

The 0.16 feature set remains: scaled dot-product and multi-head attention,
trainable composed Conv3D, 3-D pooling, functional-model checkpointing,
gradient accumulation, AUC/macro metrics, tensor mini-batches, and native LR
schedules.

## API notes

`tf.layers.transformer_block` requires `input_shape = { time, features }` and
accepts `num_heads`, `key_dim`, `ff_dim`, and `causal`. It is pre-norm: the
attention and feed-forward residual paths are each normalized.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.17.0-1.rockspec
```
