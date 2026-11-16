# Changelog

## Unreleased

- Conv2D and MaxPool2D layers;
- Conv2D/MaxPool stacks in Sequential;
- integer-list graph attributes.

## 0.4.0

- stabilized public API with an explicit surface regression test;
- CI LuaRocks installation and load check;
- pinned libtensorflow 2.18 archive checksum;
- `predict_proba`, class-weighted sparse losses, Flatten/Embedding stacks,
  weight snapshots, and parameter summaries;
- Dataset `map`, `filter`, `drop_last`, `repeat_times`, and `take`;
- CenteredRMSProp, Adadelta, Ftrl, ProximalAdagrad, and Adam clipping/weight
  decay;
- in-memory EarlyStopping restoration, resumable optimizer/epoch training
  state, runtime train/eval Dropout in Sequential, and broader leak/GC stress
  coverage;
- published `0.4.0-1` to LuaRocks;
- README and API navigation refresh;
- Sequential inference/training benchmarks and Ubuntu 22.04/24.04 CI.

## 0.3.0

- experimental eager execution (`tf.eager.Context`);
- Dense, Sequential, loss, metric, OneHot, cast, concat, and reduction
  helpers;
- sparse-label softmax cross-entropy and training, mean absolute error, and
  binary accuracy;
- EarlyStopping and ModelCheckpoint callbacks;
- reusable native V2 `Saver` and Dataset batching/`fit_dataset`;
- Saver, dataset, callback, and benchmark tooling;
- leaky ReLU and ELU activations;
- AdamW decoupled weight decay, Adam clipvalue, clip_by_value, Adadelta, and
  Ftrl;
- Embedding, Flatten, and Dropout layers;
- Sequential parameter counts, summaries, `predict_classes`, and
  `validation_split`;
- Dataset `repeat_times`, `take`, `drop_last`, `map`, and `filter`;
- Sequential `get_weights` / `set_weights`, `predict_proba`, Flatten layers,
  and Embedding/integer-input stacks with automatic int32 inputs;
- class-weighted sparse classification losses;
- fixed Dense biases to have one element per output unit;
- GC/leak stress coverage across tensors, graphs, sessions, and readers;
- public API surface test and CI LuaRocks installation job;
- pinned libtensorflow 2.18 archive checksum;
- Sequential `fit`, `evaluate`, `predict`, validation loss, callbacks, save,
  load, and GraphDef export;
- string, complex64/128, float16, bfloat16, and quantized tensors;
- TensorFlow V2 checkpoint save/restore via `SaveV2`/`RestoreV2`;
- finalizer, malformed-input, multi-output SavedModel, and dtype coverage.

## 0.2.0

- generic graph builder with typed attributes and control inputs;
- symbolic gradients, variables, initializers, and constants;
- gradient descent, Adam, RMSProp, Adagrad, and momentum optimizers;
- portable checkpoints and official `CheckpointReader`;
- native TensorFlow checkpoints;
- typed ops and `tf.nn` activations;
- LuaRocks packaging and architecture-aware libtensorflow bootstrap.

## 0.1.0

- LuaJIT FFI binding to the TensorFlow C API;
- dense tensors, GraphDef import/export, sessions, feeds/fetches/targets;
- SavedModel loading and signature decoding;
- multi-output SavedModel support.
