# Design notes

## Layers

```text
tensorflow/init.lua       public module
tensorflow/capi.lua       FFI declarations, library loading, status helpers
tensorflow/tensor.lua     TF_Tensor <-> Lua, dtype codecs
tensorflow/graph.lua      TF_Graph, GraphDef import/export
tensorflow/graph_builder  TF_OperationDescription construction
tensorflow/session.lua    TF_Session, feeds/fetches, SavedModel
tensorflow/train.lua      variables, gradients, optimizers, SaveV2/RestoreV2
tensorflow/checkpoint*    portable checkpoints and TF checkpoint reader
tensorflow/eager.lua      experimental TFE context/op/tensor handles
tensorflow/{ops,layers,losses,metrics,model}.lua  higher-level graph helpers
```

## Ownership

LuaJIT does not run `__gc` on ordinary tables, so every C handle is wrapped in
an `ffi.gc` cdata finalizer:

- `TF_Tensor`: tensor wrapper owns its handle; `Tensor:destroy()` releases it.
- `TF_Graph`: graph wrapper owns its handle; `Graph:delete()` releases it.
- `TF_Session`: session finalizer closes and deletes; `Session:close()` is
  idempotent. The finalizer closure retains the graph handle so a graph cannot
  be deleted under a live session.
- `TF_CheckpointReader` and `TFE_Context`/`TFE_TensorHandle` follow the same
  rule. Eager tensors expose idempotent `destroy()`, and eager contexts are
  reference-counted by live tensor handles and delete only after the last one
  drains.

C API calls return status through `TF_Status`. `capi.check` reads the message,
deletes the status, and raises a Lua error. Never reuse a status after checking
it.

## Tensor representation

Data is row-major. Numeric tensors use FFI ctype casts. Strings use
`TF_TString`; half/bfloat16 use bit-level conversions; complex values are
interleaved real/imaginary pairs. `totable()` returns Lua numbers, booleans,
strings, or `tf.complex` objects.

## Checkpoints

Two formats are supported:

1. the portable tensorflow-lua format (`save_checkpoint`/`load_checkpoint`),
   with a magic header, dtype, shape, and raw tensor bytes;
2. native TensorFlow V2 checkpoints through the `SaveV2` and `RestoreV2` graph
   ops, readable by `tf.CheckpointReader`.

## Eager mode

`tf.eager` is experimental and uses the `TFE_*` C API. It supports constants,
generic `execute`, arithmetic, casts, rank/shape transforms, concatenation,
gathering, slicing, padding, tiling, selection, reversal, split, reductions,
argmax, activations, and reshape. It intentionally does not provide autograd.
