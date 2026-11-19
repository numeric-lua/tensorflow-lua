# Changelog

## 2.0.0

- portable deployment bundles: GraphDef + native V2 checkpoint + manifest;
- `tf.save_bundle`/`tf.load_bundle` and Sequential/Functional
  `save_bundle` methods;
- bundle loading restores variables by name and returns a runnable
  graph/session;
- 175 dependency-free tests.

## 1.0.0

- public API stability policy and documented public surface;
- semantic-versioning and deprecation rules;
- platform limits documented explicitly;
- 174 dependency-free tests.

## 0.20.0

- unrolled LSTM and GRU layers for fixed-length `[batch, time, input]`
  sequences, with trainable kernels/recurrent/bias parameters;
- 174 dependency-free tests.

## 0.19.0

- trainable UpSampling3D for NDHWC volumes using reshape/tile/reshape;
- Sequential 3-D upsampling shape tracking and summaries;
- 173 dependency-free tests.

## 0.18.0

- tensor-backed `Dataset.from_tensor`, byte-level batching, shuffle, and
  `each` iteration;
- `Sequential:fit_dataset`/`evaluate_dataset`/`predict_dataset` integration,
  plus `predict_tensor`;
- 171 dependency-free tests.

## 0.17.0

- sinusoidal positional encoding;
- pre-norm transformer block with multi-head attention, residual connections,
  layer normalization, and a feed-forward sublayer;
- 170 dependency-free tests.

## 0.16.0

- scaled dot-product attention and multi-head attention layers with optional
  causal masking and per-head weights;
- 169 dependency-free tests.

## 0.15.0

- trainable Conv3D composed from depth slices and Conv2D kernels, supporting
  SAME/VALID padding, strides, and dilation;
- 166 dependency-free tests.

## 0.14.0

- Conv3D, MaxPool3D, AvgPool3D, GlobalAvgPool3D, and GlobalMaxPool3D layers;
- NDHWC shape tracking and summaries for 3-D Sequential stacks;
- `optimizer = "none"` for inference-only Sequential models;
- 165 dependency-free tests.

## 0.13.0

- `tf.Functional` native checkpoint save/load, weight get/set, GraphDef
  export, summaries, callbacks, validation loss, and per-output loss weights;
- 163 dependency-free tests.

## 0.12.0

- graph-level SGD/Adam/... gradient accumulation via
  `gradient_accumulation_steps`;
- macro multiclass precision/recall/F1 for sparse and one-hot labels;
- binary ROC AUC computed with a pairwise comparison matrix;
- Sequential metric integration for AUC and macro metrics;
- 162 dependency-free tests.

## 0.11.0

- tensor-backed mini-batch `fit_tensor`/`evaluate_tensor` with slicing and
  shuffled row selection;
- `tf.Functional` multi-input/multi-output graph model with custom losses and
  optimizers;
- MaxPool1D, AvgPool1D, GlobalAvgPool1D, SeparableConv2D,
  LayerNormalization, and Embedding `padding_idx`;
- Huber, hinge, and categorical hinge losses; dense class weights;
- precision, recall, and F1 metrics;
- exponential, cosine, polynomial, piecewise, and constant learning-rate
  schedules;
- Conv2DTranspose layer support in the layer API and Sequential models,
  including NHWC/NCHW graph construction, optional output shapes, training,
  and summaries.

## 0.10.0

- `Sequential:fit_tensor` and `evaluate_tensor` for full-batch Tensor feeds;
- tensor training avoids Lua-table conversion while retaining callbacks and
  BatchNorm updates.

## 0.9.0

- Sequential architecture configuration (`get_config`, `save_config`,
  `load_config`, `from_config`);
- Lua-literal model configuration serializer with cycle/function rejection.

## 0.8.0

- Reshape, Permute, UpSampling2D, ZeroPadding2D, DepthwiseConv2D, and
  GlobalMaxPool1D layers;
- depthwise convolution support in Sequential;
- global-norm gradient clipping for every optimizer.

## 0.7.0

- feedable learning rates and `ReduceLROnPlateau`;
- training-state `ModelCheckpoint` support.

## 0.6.0

- Conv1D, AveragePool2D, GlobalAveragePool2D, GlobalMaxPool2D, and Activation
  layers;
- rank-3 BatchNormalization and Conv1D+BatchNorm stack coverage;
- average-pooling stacks in Sequential;
- top-k accuracy metrics;
- per-sample weights for dense losses.

## 0.5.0

- Conv2D, MaxPool2D, and BatchNormalization layers;
- Conv2D/MaxPool/BatchNorm stacks in Sequential;
- label smoothing and per-sample weights for sparse classification losses;
- resumable BatchNorm moving statistics;
- CI coverage for libtensorflow 2.16.2/2.18.0 on Ubuntu 22.04/24.04;
- Conv2D + BatchNorm + Dense stack coverage;
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
