# Changelog

## Unreleased

- decode scalar tensor attributes through `Graph:attribute` and include them in
  `Graph:attributes`;
- add `Graph:node_def`/`Graph:op_def` for exporting operation NodeDef and OpDef
  bytes, add `Graph:version_def`, `Graph:attribute_proto`, and
  `Graph:operation_device`/`Graph:all_inputs`;
- add `Graph:debug_string` for human-readable graph dumps and decode reference
  dtypes in `Graph:operation_info`;
- add graph `tf.norm`, `tf.matrix_diag`, `tf.matrix_diag_part`,
  `tf.matrix_band_part`, `tf.matrix_set_diag`, `tf.matrix_inverse`,
  `tf.matrix_determinant`, `tf.matrix_solve`, `tf.broadcast_to`, `tf.roll`,
  `tf.zeros`, and `tf.ones`, plus eager equivalents;
- add `tf.train.exponential_moving_average` with synced shadow variables and
  update targets;
- add `tf.train.sgd` and `tf.train.optimizer` for option-table and named
  optimizer construction, with `tf.train.sgd` routing momentum/nesterov options
  through the momentum optimizer;
- add GraphDef import `default_device`, `uniquify_names`, and
  `uniquify_prefix` options;
- add graph `tf.top_k`, `tf.in_top_k`, `tf.space_to_batch_nd`, and
  `tf.batch_to_space_nd`;
- add eager `Context:top_k`, `in_top_k`, `space_to_batch_nd`, and
  `batch_to_space_nd`;
- validate portable checkpoint tensor shapes during `tf.load_checkpoint`;
- add graph `tf.tensor_scatter_nd_sub`, `tf.tensor_scatter_nd_min`,
  `tf.tensor_scatter_nd_max`, and `tf.sequence_mask`;
- add eager `Context:tensor_scatter_nd_sub`, `tensor_scatter_nd_min`,
  `tensor_scatter_nd_max`, and `sequence_mask`;
- add `tf.checkpoint_metadata` for portable checkpoint dtype/shape/byte
  metadata;
- add graph `tf.boolean_mask`, `tf.depth_to_space`, and `tf.space_to_depth`;
- add eager `Context:boolean_mask`, `depth_to_space`, and `space_to_depth`;
- add `tf.train.Saver:size` and reject missing or duplicate Saver variable
  names;
- add graph `tf.nn.softsign`, `tf.nn.selu`, `tf.invert_permutation`, and
  `tf.reverse_sequence`;
- add eager `Context:softsign`, `selu`, `invert_permutation`, and
  `reverse_sequence`;
- add graph `tf.gather_nd`, `tf.scatter_nd`, `tf.tensor_scatter_nd_update`,
  `tf.tensor_scatter_nd_add`, `tf.unique`, `tf.unique_with_counts`, `tf.diag`,
  and `tf.diag_part`;
- add eager `Context:gather_nd`, `scatter_nd`, `tensor_scatter_nd_update`,
  `tensor_scatter_nd_add`, `unique`, `unique_with_counts`, `diag`, and
  `diag_part`;
- add decoupled `weight_decay`/`wd` support to `tf.train.gradient_descent`,
  `tf.train.momentum`, `tf.train.adamax`, `tf.train.rmsprop`,
  `tf.train.centered_rmsprop`, `tf.train.adagrad`, `tf.train.adadelta`,
  `tf.train.proximal_adagrad`, and `tf.train.ftrl`;
- 271 dependency-free tests.

## 2.8.0

- add `Graph:input_list_length` and `Graph:output_list_length` for operations
  with list-valued inputs or outputs;
- add graph `tf.zeros_like`, `tf.ones_like`, `tf.identity`,
  `tf.stop_gradient`, `tf.erf`, `tf.erfc`, `tf.is_nan`, `tf.is_inf`, and
  `tf.is_finite`;
- add eager `Context:zeros_like`, `ones_like`, `identity`, `stop_gradient`,
  `erf`, `erfc`, `is_nan`, `is_inf`, and `is_finite`;
- add `CheckpointReader:dtype`, `CheckpointReader:shape`, and
  `CheckpointReader:info` for checkpoint variable metadata;
- 229 dependency-free tests.

## 2.7.0

- add `Graph:attributes` for returning decoded operation attribute tables;
- add graph `tf.floor`, `tf.ceil`, `tf.round`, `tf.sign`, `tf.rsqrt`,
  `tf.reciprocal`, `tf.reduce_variance`, and `tf.reduce_stddev`;
- add eager `Context:floor`, `ceil`, `round`, `sign`, `rsqrt`, `reciprocal`,
  `reduce_variance`, and `reduce_stddev`;
- add `clipvalue`/`clip_value` gradient clipping to RMSProp,
  CenteredRMSProp, Adagrad, Adadelta, ProximalAdagrad, and Ftrl;
- 224 dependency-free tests.

## 2.6.0

- add `Graph:set_shape` for merging refined tensor dimensions into a graph;
- add graph `tf.batch_matmul` with `adj_x`/`adj_y` support, `tf.nn.log_softmax`,
  and `tf.fill`/`tf.range`/`tf.linspace`;
- add eager `Context:batch_matmul`, `log_softmax`, `fill`, `range`, and
  `linspace`;
- add Adam Nesterov momentum (`nesterov = true`) and validate `clipvalue`;
- 219 dependency-free tests.

## 2.5.0

- add `Graph:attribute_names` for enumerating an operation's attribute names;
- add graph `tf.argmin`, `tf.nn.relu6`, `tf.add_n`, and numerically stable
  `tf.reduce_logsumexp` with `keep_dims` support;
- add eager `Context:argmin`, `Context:relu6`, `Context:add_n`, and
  `Context:reduce_logsumexp`;
- add Nesterov momentum (`nesterov = true`) and `clipvalue` gradient clipping
  to `tf.train.momentum`;
- validate integer axes in graph and eager `argmax`/`argmin`;
- 214 dependency-free tests.

## 2.4.0

- add `Graph:attribute_info` and `Graph:attribute` for typed operation
  attribute metadata and scalar/list values, including string, int, float,
  bool, type, and shape attributes;
- add graph `tf.equal`/`not_equal`/`greater`/`greater_equal`/`less`/
  `less_equal`, `tf.logical_and`/`logical_or`/`logical_xor`/`logical_not`,
  `tf.squared_difference`, `tf.reduce_prod`, and `tf.reduce_all`/`reduce_any`;
- add eager `stack`, `unstack`, `one_hot`, `clip_by_value`, comparisons,
  logical operations, boolean reductions, and `maximum`/`minimum`;
- add reusable `tf.train.clip_by_norm` and `tf.train.clip_by_global_norm`
  gradient clipping helpers, and route optimizer `clipnorm` through them;
- 209 dependency-free tests.

## 2.3.0

- add `return_elements` to `Graph:import_graph_def`/`import_file`, returning
  selected imported outputs with dtype and inferred shape;
- add `Graph:consumers`, `Graph:control_inputs`, and `Graph:control_outputs`,
  and expose consumer operations plus control edges in `operation_info`;
- expand eager execution with `concat`, `gather`, `slice`, `pad`, `tile`,
  `where`, `reverse`, and `split`, including string and multi-output op
  attributes;
- add the `tf.train.adamax` optimizer with first-moment, infinity-norm, and
  beta1-power state;
- add graph `tf.stack`, `tf.unstack`, `tf.slice`, and `tf.reverse` ops;
- 203 dependency-free tests.

## 2.2.0

- map imported GraphDef inputs to existing graph outputs through
  `Graph:import_graph_def`/`import_file` `input_map` options;
- add `Graph:operations`, `Graph:shape`, and `Graph:operation_info` for
  operation, consumer, dtype, and inferred-shape introspection;
- expand eager execution with mean/max/min/prod reductions, keep-dims support,
  `exp`, `log`, `sqrt`, `abs`, `tanh`, `softplus`, `softmax`, `elu`, and
  `leaky_relu`;
- add graph `tf.split` (Split/SplitV), `tf.pad` (Pad/MirrorPad), `tf.tile`, and
  `tf.where` (SelectV2) ops;
- support custom upstream seeds through `tf.gradients(..., grad_ys)` and sum
  gradients from multiple outputs;
- 196 dependency-free tests.

## 2.1.0

- prefix-aware `Graph:import_graph_def`/`import_file` options for importing a
  GraphDef into a namespace more than once;
- eager tensor `destroy()`, finalizer release counters, and context deletion
  deferred until all live eager handles drain;
- eager `div`, `cast`, `transpose`, `expand_dims`, `squeeze`, `reduce_sum`,
  and `argmax`, plus integer-list attributes;
- `tf.train.gradient_descent` exposes `clipnorm`, `clipvalue`, and precomputed
  gradients through an options table;
- graph `tf.transpose`, `tf.expand_dims`, `tf.squeeze`, and `tf.gather` ops;
- include `tensorflow.bundle` in the SCM and versioned rockspecs so installed
  packages load successfully;
- verify the full suite against libtensorflow 2.16.2, 2.18.0, and 2.18.1;
- pin Linux x86-64 archive checksums for 2.16.2 and 2.18.1 and reject a
  destination containing a different libtensorflow version;
- report libtensorflow 2.19/2.20 as unavailable with a clear diagnostic
  instead of attempting archives that Google never published;
- add a compatibility matrix with operation coverage for separable
  convolution, pooling, and checkpoints;
- extend CI to 2.18.1 and add fetch-script version regression tests;
- 187 dependency-free tests.

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
