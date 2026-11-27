# tensorflow-lua API

## Contents

- [Loading](#loading)
- [Tensors](#tensors)
- [Graphs](#graphs)
- [Losses](#losses)
- [Data batching](#data-batching)
- [Callbacks](#callbacks)
- [Metrics](#metrics)
- [Layers](#layers)
- [Sequential model](#sequential-model)
- [Eager execution (experimental)](#eager-execution-experimental)
- [Training examples](#training-examples)
- [Typed ops](#typed-ops)
- [Sessions](#sessions)
- [Errors](#errors)

## Loading

```lua
local tf = require("tensorflow")
print(tf.version())          -- "2.18.0"
```

The module is available from LuaRocks as `tensorflow-lua` and has been tested
with libtensorflow 2.16.2, 2.18.0, and 2.18.1; see
[docs/compatibility.md](compatibility.md). It loads `libtensorflow.so` from
`TENSORFLOW_LIB`, `TENSORFLOW_ROOT`, `third_party/libtensorflow`, or the dynamic
linker path.

`tf.VERSION` is the binding version (`2.8.0`); `tf.version()` returns
libtensorflow's version.

## Tensors

### `tf.tensor(value[, opts])`

Converts a scalar or rectangular nested Lua table to a `tf.Tensor` with a
TF_Tensor backing. `opts.dtype` defaults to `float32` for numeric tables. Strings, booleans, and
complex values infer `string`, `bool`, and `complex64`. Numeric dtypes are
converted explicitly; no implicit dtype promotion occurs. `string` tensors accept Lua strings and use
TensorFlow's `TF_TString` layout; raw-byte export is intentionally unsupported
for strings. Quantized `qint8`, `quint8`, `qint16`, `quint16`, and `qint32`
values are stored as their integer bit patterns. Complex tensors use
`tf.complex(real, imag)` values and `totable()` returns objects with `real` and
`imag` fields. `float16` and `bfloat16` tensors convert from and to Lua numbers;
`half` is an alias for `float16`.

```lua
local a = tf.tensor({ { 1, 2 }, { 3, 4 } })
local b = tf.tensor({ 1, 2 }, { dtype = "int64" })
```

### Methods

| Method | Result |
| --- | --- |
| `t:shape()` / `t:sizes()` | Lua array of dimensions |
| `t:dtype()` | dtype name |
| `t:dim()` | rank |
| `t:numel()` | element count |
| `t:nbytes()` | backing byte count |
| `t:totable()` | scalar or nested Lua table |
| `t:bytes()` | raw tensor bytes |
| `t:handle()` | raw `TF_Tensor*` (advanced) |
| `tostring(t)` | shape/dtype summary |

Tensors own their output buffers through `ffi.gc` (which runs on LuaJIT) and
also expose `t:destroy()` for deterministic cleanup.

### `tensor:slice(start, stop)`

Returns rows `start..stop` along the first dimension as a new contiguous tensor.
The copy is byte-level and does not add a graph operation. `tensor:take_rows(indices)`
selects an arbitrary 1-based row order.

## Graphs

### `tf.Graph.new()`

Creates a TF_Graph.

### `graph:operations()`

Returns the graph's operation names in TensorFlow's iteration order.

### `graph:node_def(output)`

Returns the operation's serialized `NodeDef` protobuf bytes. The argument may be
an operation name, `"op:index"`, or a graph output. TensorFlow status errors and
missing operations become Lua errors.

```lua
local bytes = graph:node_def("logits")
```

### `graph:op_def(op_type)`

Returns the registered operation's serialized `OpDef` protobuf bytes for
`op_type`. Missing or unregistered op types raise TensorFlow status errors as
Lua errors.

```lua
local schema = graph:op_def("AddV2")
```

### `graph:version_def()`

Returns the graph's serialized `VersionDef` protobuf bytes through
`TF_GraphVersions`, with `TF_Status` error handling and buffer cleanup.

### `graph:version_info()`

Decodes `Graph:version_def()` into `{ producer, min_consumer, bad_consumers }`.

### `graph:debug_string()`

Returns the graph in TensorFlow's human-readable debug format. The format is
not stable across TensorFlow releases and is intended for inspection.

### `graph:operation_device(output)` / `graph:all_inputs(output)`

`operation_device` returns the device string assigned to an operation, or `nil`
when no device was assigned. `all_inputs` returns the flattened input edges as
`{ input, name, index, dtype }` records, including every tensor in list-valued
inputs such as `Pack` and `AddN`.

### `graph:shape(output)`

Returns the inferred shape of an output (a graph output table, `"op"`, or
`"op:index"`), or `nil` when the rank is unknown. Unknown dimensions are `-1`.
Invalid output indices raise a Lua error before reaching TensorFlow.

### `graph:set_shape(output, shape)`

Merges a refined shape into an operation output and returns the graph. `shape`
is a table of integer dimensions, where `-1` keeps a dimension unknown, or
`nil` for unknown rank. TensorFlow rejects incompatible merges through
`TF_Status`, so an existing `{ -1, 2 }` can become `{ 3, 2 }` but not `{ 4, 2 }`.

```lua
graph:set_shape("images:0", { 8, 224, 224, 3 })
```

### `graph:operation_info(output)`

Returns `{ name, op_type, inputs, outputs, control_inputs, control_outputs }`
for an operation or output. Each input has `input`, `name`, `output`, and
`dtype`; each output has `name`, `index`, `dtype`, `shape` (as returned by
`graph:shape`), `consumers` (a count), and `consumer_ops` (operation names).
Reference dtypes are normalized to their value dtype names.
`control_inputs` and `control_outputs` list the control-edge operation names.

### `graph:attribute_info(output, name)` / `graph:attribute(output, name)`

Reads a named operation attribute directly from a `TF_Operation`.
`attribute_info` returns `{ name, operation, kind, is_list, list_size,
total_size }`, where `kind` is one of `string`, `int`, `float`, `bool`, `type`,
`shape`, `tensor`, `placeholder`, or `func`. `attribute` decodes scalar and
list values for string, int, float, bool, type, and shape attributes, and
returns scalar tensor attributes as `tf.Tensor` objects (tensor-list,
placeholder, and function attributes are not decoded). Dtype attributes are
returned as documented dtype names and shape attributes as tables of
dimensions.

```lua
local info = graph:attribute_info("logits", "T")   -- kind = "type"
local dtype = graph:attribute("logits", "T")      -- "float32"
local dims = graph:attribute("images", "shape")    -- { 2, 3 }
```

### `graph:attribute_names(output)`

Returns the operation's attribute names in TensorFlow's attribute order. The
argument may be an operation name, `"op:index"`, or a graph output. Missing
operations raise a Lua error.

```lua
local names = graph:attribute_names("logits")  -- e.g. { "T", "shape" }
```

### `graph:attributes(output)`

Returns a list of decoded attribute records in attribute-name order. Each record
contains the fields from `attribute_info` plus `value` for string, int, float,
bool, type, shape, and scalar tensor attributes. Tensor-list, placeholder, and
function attributes are returned without `value`. Tensor values returned by
`attribute`/`attributes` are owned by Lua and should be destroyed explicitly
when no longer needed.

```lua
for _, attribute in ipairs(graph:attributes("logits")) do
  print(attribute.name, attribute.kind, attribute.value)
end
```

### `graph:attribute_proto(output, name)`

Returns the raw serialized `AttrValue` protobuf bytes for an operation
attribute, including tensor-list, placeholder, and function kinds that
`graph:attribute` does not decode. Missing attributes raise TensorFlow status
errors as Lua errors.

### `graph:input_list_length(output, name)` / `graph:output_list_length(output, name)`

Return the number of tensors in a list-valued operation argument. `Pack`/`AddN`
inputs and `Split`/`Unpack` outputs are the typical uses. Missing arguments and
non-list arguments raise TensorFlow status errors as Lua errors.

```lua
local count = graph:input_list_length("packed", "values")
local splits = graph:output_list_length("split", "output")
```

### `graph:consumers(output)`

Returns the operations that consume an output as a list of
`{ name, input }`, where `input` is the consumer's input index. Invalid output
indices raise a Lua error.

### `graph:control_inputs(operation)` / `graph:control_outputs(operation)`

Return sorted-by-graph-order lists of operation names connected by control
edges. The argument may be an operation name, `"op:index"`, or a graph output.

### `graph:import_graph_def(bytes[, opts])`

Imports a serialized `GraphDef` protobuf from a Lua string. `opts.prefix`
places every imported operation in a namespace, so the same GraphDef can be
imported more than once into one graph:

```lua
graph:import_graph_def(bytes, { prefix = "left" })
graph:import_graph_def(bytes, { prefix = "right" })
local z = session:run({ ["left/x"] = 1, ["left/y"] = 2 }, "left/z:0")
```

`opts.input_map` maps imported GraphDef input names (`"x"` or `"x:0"`) to
existing graph outputs (a graph output table or an `"op:index"` string):

```lua
local replacement = tf.constant(graph, { 1, 2 }, "replacement")
graph:import_graph_def(bytes, { input_map = { x = replacement } })
```

The imported graph then uses `replacement` instead of a placeholder for `x`.

`opts.return_elements` selects outputs from the imported graph and returns them
as graph outputs instead of the graph itself. Names are resolved after import,
so include any `prefix`:

```lua
local outputs = graph:import_graph_def(bytes, {
  prefix = "left",
  return_elements = { "left/z:0" },
})
local value = session:run({ ["left/x"] = 1, ["left/y"] = 2 }, outputs)[1]
```

Each returned entry has `oper`, `index`, `name`, `dtype`, and `shape`.

`opts.default_device` applies a device to imported operations.
`opts.uniquify_names` renames conflicting operations, and
`opts.uniquify_prefix` uniquifies the supplied namespace prefix so the same
GraphDef can be imported repeatedly without manually changing names.

### `graph:import_file(path[, opts])`

Reads and imports a `.pb` file, accepting the same options as
`import_graph_def`.

### `graph:new_operation(op_type, name)`

Creates an operation description for the generic graph builder. Description
methods return the description and must be finished:

```lua
local desc = graph:new_operation("AddV2", "z")
desc:add_input("x:0")
desc:add_input("y:0")
desc:set_type("T", "float32")
local z = desc:finish()       -- { oper=..., index=0, name="z:0" }
```

Description methods: `add_input`, `add_inputs`, `add_control_input`,
`set_device`, `set_type`, `set_int`, `set_float`, `set_bool`, `set_string`,
`set_shape`, `set_tensor`, and `finish`.

### `tf.placeholder(graph, dtype, shape, name)`

Creates a placeholder output. `shape` may be `nil` for unknown rank or a Lua
array; `-1` denotes an unknown dimension.

### `tf.add(graph, a, b, name[, dtype])`

### `tf.matmul(graph, a, b, name[, dtype])`

### `tf.sub`, `tf.mul`, `tf.square`

Create `AddV2`, `MatMul`, `Sub`, `Mul`, and `Square` operations. `dtype`
defaults to `float32`.

### `tf.constant(graph, value, name[, dtype])`

Creates a Const output. Values may be Lua scalars/nested tables or an existing
`tf.Tensor`.

### `tf.variable(graph, initial, name[, opts])`

Creates a `VariableV2`, its `Assign` initializer, and an `Identity` read output.
Returns a table with `ref`, `read`, `initializer`, `dtype`, and `shape`.
Supported options: `dtype`, `container`, `shared_name`.

### `tf.train.gradient_descent(graph, loss, variables, lr[, name[, opts]])`

Adds `ApplyGradientDescent` targets for the variables and returns
`targets, gradients, learning_rate`. Initialize variables first with
`session:run({}, {}, { variable.initializer })`, then run the target list once
per training step. Options: `clipnorm` limits the global gradient norm,
`clipvalue` limits each gradient element, `weight_decay`/`wd` applies decoupled
weight decay, and `gradients` supplies precomputed gradients instead of calling
`tf.gradients`. Clipping and decay bounds must be non-negative (clipping bounds
positive). Every optimizer documented here accepts the same
`clipnorm`/`clip_norm`, `clipvalue`/`clip_value`, and `weight_decay`/`wd`
options.

### `tf.train.sgd(graph, loss, variables[, opts])`

Convenience SGD entry point accepting `learning_rate`/`lr` (default `0.01`),
`name`, and the same clipping/weight-decay options as `gradient_descent`.
Supplying `momentum`/`nesterov` routes through `tf.train.momentum`.

### `tf.train.optimizer(kind, graph, loss, variables[, opts])`

Builds an optimizer from a case-insensitive name. Supported kinds are `sgd`/
`gradient_descent`, `adam`, `adamw`, `adamax`, `rmsprop`, `centered_rmsprop`,
`adagrad`, `adadelta`, `ftrl`, `proximal_adagrad`, and `momentum`. Unknown names
raise a Lua error.

### `tf.train.exponential_moving_average(graph, variables, decay[, opts])`

Creates shadow variables for each variable, `initializers` that zero and then
sync each shadow from its variable, and `updates` targets that apply
`shadow = shadow * decay + variable * (1 - decay)`. `decay` must be between 0
and 1. Run variable initializers before `initializers`.

### `tf.train.gradient_accumulation(graph, variables, gradients, steps[, opts])`

Creates accumulator variables initialized to zero plus `accumulate`, `reset`, and
`gradients` targets. `gradients` exposes the accumulated gradients divided by
`steps`, suitable for an optimizer's `gradients` option. `steps` must be a
positive integer and `gradients` must match `variables` in length.

### `tf.train.clip_by_norm(graph, gradient, clip_norm[, name[, dtype]])`

Returns `clipped_gradient, gradient_norm`, scaling a single graph-output
gradient so its L2 norm is at most `clip_norm`. `clip_norm` must be positive.

### `tf.train.clip_by_global_norm(graph, gradients, clip_norm[, name[, dtype]])`

Returns `clipped_gradients, global_norm` for a non-empty list of graph-output
gradients, scaling all gradients by the same factor when their combined L2 norm
exceeds `clip_norm`. The returned norm is the pre-clipping norm.

### `tf.train.adam(graph, loss, variables[, opts])`

Adds `ApplyAdam` updates and slot variables. Options: `learning_rate`/`lr`
(default `0.001`), `beta1` (0.9), `beta2` (0.999), `epsilon` (1e-8), `name`,
`nesterov` (or `use_nesterov`), `weight_decay`, and `clipvalue`/`clip_value`
(gradient-value clamping, must be positive). Returns
`targets, gradients, state`.
Initialize each variable and every entry in `state.initializers` before
training.

### `tf.train.adamw(graph, loss, variables[, opts])`

Adam with decoupled weight decay; defaults `weight_decay = 0.01`. Accepts the
same options and returns the same triple as `tf.train.adam`.

### `tf.train.adamax(graph, loss, variables[, opts])`

Adds `ApplyAdaMax` updates with first-moment, infinity-norm, and
`beta1_power` state. Options: `learning_rate`/`lr` (default `0.001`), `beta1`
(0.9), `beta2` (0.999), `epsilon` (1e-8), `weight_decay`/`wd`, `clipvalue`, and
`name`. Returns
`targets, gradients, state`; initialize every `state.initializers` entry
before training. The state table also exposes `state.beta1_power`.

### `tf.train.rmsprop(graph, loss, variables[, opts])`

Adds `ApplyRMSProp` updates and slot variables. Options: `learning_rate`/`lr`
(default `0.001`), `rho` (0.9), `momentum` (0), `epsilon` (1e-10),
`weight_decay`/`wd`, `centered` (route to centered RMSProp), `name`. Returns
`targets, gradients, state`; initialize `state.initializers`.

### `tf.train.Saver.new(graph, variables)`

Reusable native V2 checkpoint saver. `variables` is a list of
`{ name = ..., variable = ... }` entries. Names must be non-empty and unique.
Methods: `save(session, prefix)`, `restore(session, prefix)`, `names()`, and
`size()`.

### `tf.train.centered_rmsprop(graph, loss, variables[, opts])`

Adds `ApplyCenteredRMSProp` updates with mean-gradient, variance, and momentum
slots. Options: `learning_rate`/`lr` (default `0.001`), `rho` (0.9),
`momentum` (0), `epsilon` (1e-10), `weight_decay`/`wd`, `name`. Initialize
`state.initializers`.

### `tf.train.adagrad(graph, loss, variables[, opts])`

Adds `ApplyAdagrad` updates and accumulator variables. Options:
`learning_rate`/`lr` (default `0.01`), `weight_decay`/`wd`, `name`. Initialize
`state.initializers`.

### `tf.train.adadelta(graph, loss, variables[, opts])`

Adds `ApplyAdadelta` updates with accumulator and update slots. Options:
`learning_rate`/`lr` (default `1.0`), `rho` (0.95), `epsilon` (1e-8),
`weight_decay`/`wd`, `name`. Initialize `state.initializers`.

### `tf.train.proximal_adagrad(graph, loss, variables[, opts])`

Adds FOBOS/Adagrad updates with an accumulator slot. Options:
`learning_rate`/`lr` (default `0.01`), `l1` (0), `l2` (0), `weight_decay`/`wd`,
`name`. Initialize `state.initializers`.

### `tf.train.ftrl(graph, loss, variables[, opts])`

Adds Ftrl-Proximal updates with accumulator and linear slots. Options:
`learning_rate`/`lr` (default `0.001`), `l1` (0), `l2` (0),
`weight_decay`/`wd`, `lr_power`
(`-0.5`), `name`. Initialize `state.initializers`.

### `tf.train.momentum(graph, loss, variables[, opts])`

Adds `ApplyMomentum` updates and accumulator variables. Options:
`learning_rate`/`lr` (default `0.01`), `momentum` (0.9), `nesterov` (or
`use_nesterov`), `weight_decay`/`wd`, `clipnorm`, `clipvalue`/`clip_value`, and
`name`. Initialize `state.initializers`.

### `tf.save_checkpoint(session, path, variables)`

Writes a portable tensorflow-lua checkpoint. `variables` is a list of
`{ name = ..., variable = ... }` entries. The format stores dtype, shape, and
raw tensor bytes; it is not TensorFlow's native checkpoint format.

### `tf.save_v2(graph, prefix, variables[, name])`

Builds a native TensorFlow `SaveV2` target. `variables` is a list of
`{ name = ..., variable = ... }` entries; values are written to the V2
checkpoint prefix (for example `model.ckpt`). Returns a graph output usable as a
session target.

### `tf.restore_v2(graph, prefix, variables[, name])`

Builds `RestoreV2` reads and `Assign` targets for a native TensorFlow V2
checkpoint. Run the returned target list after creating the session.

### `tf.load_checkpoint(graph, path, variables)`

Builds `Assign` targets from a portable checkpoint saved by
`tf.save_checkpoint`. Returns the target list for `session:run`. Variable dtype
and shape metadata is validated during graph construction.

### `tf.checkpoint_metadata(path)`

Reads a portable tensorflow-lua checkpoint without constructing graph
operations and returns a map of variable names to `{ dtype, shape, bytes }`
metadata. Missing or corrupt files raise Lua errors.

### `tf.CheckpointReader`

Reads official TensorFlow checkpoint files:

```lua
local reader = tf.CheckpointReader.new("model.ckpt")
print(reader:size())
for _, name in ipairs(reader:variables()) do print(name) end
local tensor = reader:tensor(name)
print(reader:dtype(name), table.concat(reader:shape(name), ","))
local info = reader:info(name)  -- { name, dtype, shape }
reader:close()
```

`dtype(name)` and `shape(name)` read checkpoint metadata without
`TF_Tensor` allocation; `info(name)` combines them. Missing variables raise a
Lua error.

## Losses

### `tf.losses.mean_squared_error(graph, predictions, targets[, opts])`

Builds a squared-error mean. Options: `axes` (default `{0}`), `keep_dims`,
`dtype`, `name`.

### `tf.losses.sparse_softmax_cross_entropy(graph, logits, labels[, opts])`

Builds sparse-label softmax cross-entropy. `labels` is an integer tensor of
class indices; options: `labels_dtype` (default `int32`), `axes` (default
`{0}`), `keep_dims`, `dtype`, `name`. The TensorFlow C API does not register a
gradient for this op; `Sequential` converts sparse labels to one-hot and uses
the differentiable dense loss instead.

### `tf.losses.mean_absolute_error(graph, predictions, targets[, opts])`

Builds an absolute-error mean. Options: `axes` (default `{0}`), `keep_dims`,
`dtype`, `name`.

### `tf.losses.binary_cross_entropy(graph, logits, labels[, opts])`

Builds a numerically stable sigmoid binary cross-entropy mean using
`relu(x) - x*y + softplus(-x)`. Options: `axes` (default `{0}`), `keep_dims`,
`dtype`, `name`.

### `tf.losses.softmax_cross_entropy(graph, logits, labels[, opts])`

Builds a `SoftmaxCrossEntropyWithLogits` mean over the batch axis (default
`axes = {0}`). Options: `axes`, `keep_dims`, `dtype`, `name`.

### `tf.losses.huber(graph, predictions, targets[, opts])`

Huber loss with `opts.delta` (default `1`). `opts.axes` defaults to `{ 0 }`.

### `tf.losses.hinge(graph, predictions, targets[, opts])`

Binary hinge loss for `-1`/`1` targets, with `opts.margin` (default `1`).

### `tf.losses.categorical_hinge(graph, logits, targets[, opts])`

Multi-class categorical hinge loss. Pass `per_example = true` to return the
per-example vector instead of the mean.

### `tf.losses.softmax_cross_entropy` class weights

`softmax_cross_entropy` accepts `opts.class_weights`, a zero-based table of
per-class multipliers. Weighted losses are normalized by the sum of the
per-example weights.

## Data batching

`tf.data.Dataset.from_tensor(x, y)` creates a tensor-backed dataset. Batches
are produced with byte-level row slices and never convert the full tensors to
Lua tables. It supports `batch(size)`, `shuffle(seed)`, `take(count)`, `each()`,
and `size()`. `Sequential:fit_dataset`, `evaluate_dataset`, and
`predict_dataset` detect the tensor dataset and route to the tensor paths;
`predict_tensor` accepts a tensor directly.


```lua
local dataset = tf.data.Dataset.new(x, y):shuffle(7):batch(32)
for batch in dataset:each() do
  -- batch.x and batch.y
end
```

`Dataset.new(x[, y])`, `size()`, `batch(size[, opts])`, `shuffle([seed])`,
`map(fn)`, `filter(fn)`, `split(fraction)`, `repeat_times(count)`,
`take(count)`, and `each()` return batches as `{ x = ..., y = ... }`. `opts.drop_last` discards a partial final
batch. (`repeat` is a Lua keyword, so the method is
`repeat_times`.)

`model:fit(dataset, opts)` trains directly from labeled dataset batches;
`fit_dataset(dataset, opts)` is an alias. `Dataset:split(fraction)` returns
train/validation datasets, and `model:evaluate_dataset` /
`model:predict_dataset` accept Dataset objects directly.

## Callbacks

### `tf.callbacks.EarlyStopping(opts)`

Stops when a monitored history entry stops improving. Options: `monitor`
(default `val_loss`), `mode` (`min`/`max`), `patience`, `min_delta`, and
`restore_best_weights`. Records `stopped_epoch`; restoration uses in-memory
weight snapshots.

### `tf.callbacks.ReduceLROnPlateau(opts)`

Multiplies `model._current_lr` when a monitored entry stops improving. Options:
`monitor` (default `val_loss`), `mode`, `factor` (default `0.1`), `patience`
(default `10`), `min_delta`, `min_lr`, and `cooldown`. Learning-rate changes are
fed into the optimizer on the next epoch.

### `tf.callbacks.ModelCheckpoint(path, opts)`

Saves the model after epochs. `path` may contain `{epoch}`. Options: `monitor`,
`mode`, `save_best_only`, `min_delta`, and `save_training` (persist optimizer
slots, BatchNorm state, epoch, and learning rate via `save_training`).

## Metrics

### `tf.metrics.binary_accuracy(graph, predictions, labels[, opts])`

Thresholds probability predictions (default `0.5`) and compares them with
`0`/`1` labels. Options: `threshold`, `dtype`, `labels_dtype`, `axes`, `name`.

### `tf.metrics.accuracy(graph, predictions, labels[, opts])`

Compares integer class predictions to labels with `Equal`, casts to float32,
and reduces the mean over the batch. Options: `dtype` (default `int32`), `axes`,
`name`.

### `tf.metrics.precision`, `recall`, `f1`

Binary metrics over probabilities (thresholded by `opts.threshold`, default
`0.5`). `opts.axes` defaults to `{ 0 }`; pass `{ 0, 1 }` for `[batch, 1]`
outputs. ### `tf.metrics.auc(graph, predictions, labels[, opts])`

Binary ROC AUC using a pairwise comparison matrix; ties count as half a win.
The implementation is exact but O(N^2) in the batch size. Use moderate
evaluation batches.

### `tf.metrics.multiclass_prf(graph, predictions, labels, num_classes[, opts])`

Returns macro precision, recall, F1, and a per-class table for integer class
predictions and labels. Classes are numbered from zero.

## Layers

### `tf.layers.conv2d(graph, input, filters, kernel_size[, opts])`

Builds a bias-enabled 2-D convolution with a learned
`[kh, kw, in_channels, filters]` kernel. Required option: `in_channels`.
Other options: `stride`, `padding` (`SAME`/`VALID`), `dilation`, `use_bias`,
`activation`, `dtype`, `name`. Returns
`{ output, weight, bias, variables, initializers, filters, kernel_size }`.

### `tf.layers.conv2d_transpose(graph, input, filters, kernel_size[, opts])`

Builds a bias-enabled transposed 2-D convolution (`Conv2DBackpropInput`) with
a learned `[kh, kw, filters, in_channels]` kernel. Required option:
`in_channels`. Other options: `stride`, `padding` (`SAME`/`VALID`), `dilation`,
`output_shape`, `data_format` (`NHWC` by default; `NCHW` is forwarded to the
graph op, although CPU TensorFlow only executes NHWC), `use_bias`, `activation`,
`dtype`, `name`. `output_shape` may be
`{ height, width, filters }` or the full `{ batch, height, width, filters }`
(NHWC); when omitted, spatial sizes are derived from `input_shape` or the
input tensor. Returns
`{ output, weight, bias, variables, initializers, filters, in_channels,
kernel_size, stride, padding, output_shape, data_format }`.

### `tf.layers.conv1d(graph, input, filters, kernel_size[, opts])`

Builds a 1-D convolution over `[batch, steps, channels]` input using an
internal 2-D convolution lift. Required option: `in_channels`. Other options:
`stride`, `padding`, `dilation`, `use_bias`, `activation`, `dtype`, `name`.
Returns `{ output, weight, bias, variables, initializers, filters,
kernel_size }`.

### `tf.layers.max_pool2d(graph, input, kernel_size[, opts])`

Builds a `MaxPool` operation. Options: `stride`, `padding` (`VALID` by
default), `name`. Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.batch_normalization(graph, input, channels[, opts])`

Builds gamma/beta variables, moving mean/variance, and runtime-conditional
normalization. Options: `training` (boolean or boolean graph output), `axes`
(default `{0}`), `epsilon` (1e-5), `momentum` (0.99), `dtype`, `name`. The
moving statistics update only while training is true. Returns
`{ output, gamma, beta, moving_mean, moving_variance, variables, initializers }`.

### `tf.layers.avg_pool2d(graph, input, kernel_size[, opts])`

Builds an `AvgPool` operation. Options: `stride`, `padding` (`VALID` by
default), `name`. Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.global_avg_pool2d(graph, input[, opts])`

Averages the H/W dimensions and squeezes them, producing `[batch, channels]`.
Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.global_max_pool2d(graph, input[, opts])`

Max-pools the H/W dimensions and squeezes them, producing
`[batch, channels]`. Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.activation(graph, input, activation[, opts])`

Applies a named or functional activation without parameters. Returns
`{ output, variables = {}, initializers = {} }`.

### `tf.layers.reshape(graph, input, shape[, opts])`

Builds a `Reshape` operation. `shape` includes the batch dimension (use `-1`
for it). Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.permute(graph, input, perm[, opts])`

Builds a `Transpose` operation. `perm` includes the batch dimension, e.g.
`{ 0, 2, 1 }`. Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.upsampling2d(graph, input, scale[, opts])`

Nearest-neighbor resizes an NHWC input. Requires `opts.input_shape = { h, w,
c }`. Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.zero_padding2d(graph, input, padding[, opts])`

Pads an NHWC input. `padding` is a number or
`{ top, bottom, left, right }`. Returns `{ output, variables = {},
initializers = {} }`.

### `tf.layers.depthwise_conv2d(graph, input, multiplier, kernel_size[, opts])`

Depthwise convolution with bias and optional activation. Options include
`in_channels`, `stride`, `padding`, `dilation`, `use_bias`, `activation`,
`dtype`, and `name`. Returns
`{ output, weight, bias, variables, initializers, channels }`.

### `tf.layers.global_max_pool1d(graph, input[, opts])`

Max-pools the time dimension of `[batch, steps, channels]`, returning
`[batch, channels]`. Returns `{ output, variables = {}, initializers = {} }`.

### `tf.layers.max_pool1d`, `avg_pool1d`

Pool a rank-3 `[batch, steps, channels]` tensor. `opts.stride` defaults to the
kernel size and `opts.padding` to `"VALID"`.

### `tf.layers.global_avg_pool1d(graph, input[, opts])`

Reduces a rank-3 `[batch, steps, channels]` tensor over the time axis.

### `tf.layers.attention(graph, query, key, value[, opts])`

Scaled dot-product attention over rank-3 `[batch, time, features]` tensors.
Requires `opts.query_shape` and `opts.key_shape` as `{ time, features }`.
Returns `context, weights`. `opts.causal` adds a lower-triangular mask;
`opts.mask` adds an explicit additive mask.

### `tf.layers.multi_head_attention(graph, query, key, value[, opts])`

Projects query/key/value, splits them into `opts.num_heads` heads of
`opts.key_dim`/`opts.value_dim`, applies scaled dot-product attention, and
projects to `opts.output_dim`. Requires `query_shape`, `key_shape`, and
optionally `value_shape`. Returns a table with `output`, `weights`, `variables`,
and `initializers`. The layer is composed from portable ops and is trainable.

### `tf.layers.positional_encoding(graph, input[, opts])`

Adds a fixed sinusoidal encoding to a `[batch, time, features]` tensor.
Requires `opts.input_shape = { time, features }`; feature count must be even.
Returns `{ output, variables, initializers }`.

### `tf.layers.transformer_block(graph, input[, opts])`

Pre-norm transformer block: multi-head self-attention + residual + layer norm,
then a two-layer feed-forward + residual + layer norm. Requires
`opts.input_shape = { time, features }`; accepts `num_heads`, `key_dim`,
`ff_dim`, and `causal`. Returns `{ output, variables, initializers }`.

### `tf.layers.upsampling3d(graph, input, scale[, opts])`

Nearest-neighbor 3-D upsampling for an NDHWC volume. `scale` is a number or
`{ d, h, w }`; requires `opts.input_shape = { d, h, w, c }`. Implemented with
reshape/tile/reshape and trainable through those ops.

### `tf.layers.lstm(graph, input[, hidden[, opts]])`

Unrolled LSTM for fixed-length `[batch, time, input]` sequences. Requires
`opts.input_shape = { time, input }` and `opts.units`. Returns `output`,
`h`, `c`, `variables`, and `initializers`. `hidden` may supply `{ h, c }`
tensors; otherwise zero states are created.

### `tf.layers.gru(graph, input[, hidden_state[, opts]])`

Unrolled GRU with input and recurrent gate projections for static-length
sequences. Requires `opts.input_shape` and `opts.units`; returns `output`, `h`,
`variables`, and `initializers`.

### `tf.layers.conv3d(graph, input, filters, kernel_size[, opts])`

NDHWC 3-D convolution with `opts.in_channels`, strides, dilation, padding,
bias, and activation. It is implemented as a depth-slice composition of
portable Conv2D kernels because the C library's native Conv3D op has no
registered gradient; it supports SAME/VALID padding, strides, and dilation and
is trainable. The tradeoff is one Conv2D per kernel depth slice.

### `tf.layers.max_pool3d`, `avg_pool3d`, `global_avg_pool3d`, `global_max_pool3d`

3-D pooling over NDHWC tensors. The global variants reduce the depth, height,
and width axes.

### `tf.layers.separable_conv2d(graph, input, filters, kernel_size[, opts])`

Depthwise + pointwise convolution. Requires `opts.in_channels` and accepts
`opts.depth_multiplier`, `opts.stride`, `opts.padding`, `opts.dilation`,
`opts.use_bias`, and `opts.activation`. The implementation composes
`DepthwiseConv2dNative` and `Conv2D` because the C library on the supported
hosts does not register the fused `SeparableConv2D` op.

### `tf.layers.layer_normalization(graph, input[, opts])`

Normalizes the last dimensions named by `opts.normalized_shape` and learns
`gamma`/`beta`. `opts.epsilon` defaults to `1e-5`.

### `tf.layers.embedding` padding

`opts.padding_idx` accepts a zero-based row index; that embedding row is
initialized to zero. Indices remain zero-based because they feed `GatherV2`
directly.

### `tf.layers.dense(graph, input, units, opts)`

Builds a Dense layer with a `[input_size, units]` weight variable and optional
bias. Required option: `input_size`. Other options: `use_bias` (true),
`activation` (`"relu"`, `"sigmoid"`, `"tanh"`, a custom function, or nil),
`dtype`, `name`.

Returns `{ output, weight, bias, variables, initializers, input_size, units }`.

### `tf.layers.dropout(graph, input, rate[, opts])`

Builds an inverted-dropout mask from the dynamic input shape. Options:
`training` (default true; false returns the input unchanged; a boolean graph
output switches at runtime), `seed`, `seed2`, `dtype`, `name`. Returns
`{ output }`.

### `tf.layers.flatten(graph, input, input_size[, opts])`

Reshapes batched input to `[-1, input_size]`. Options: `name`, `dtype`.
Returns `{ output, input_size, variables = {}, initializers = {} }`.

### `tf.layers.embedding(graph, input, input_dim, output_dim[, opts])`

Builds an embedding table and gathers `input` indices. Options: `indices_dtype`
(default `int32`), `dtype`, `name`. Returns
`{ output, weight, variables, initializers, input_dim, output_dim }`.

### `tf.layers.sequential(graph, input, definitions, input_size)`

Chains Dense layers. Each definition is `{ units = n, options = {...} }`.
Returns the layer list. The caller must run every layer initializer.

## Sequential model

```lua
local model = tf.Sequential.new({ input_shape = { 2 } })
model:add({ units = 4, options = { activation = "relu" } })
model:add({ type = "conv1d", filters = 8, kernel_size = 3,
  options = { padding = "SAME", activation = "relu" } })
model:add({ type = "conv2d", filters = 8, kernel_size = 3,
  options = { padding = "SAME", activation = "relu" } })
model:add({ type = "conv2d_transpose", filters = 8, kernel_size = 3,
  stride = 2, options = { padding = "SAME", activation = "relu" } })
model:add({ type = "max_pool2d", kernel_size = 2, stride = 2 })
model:add({ type = "avg_pool2d", kernel_size = 2, stride = 2 })
model:add({ type = "global_avg_pool2d" })
model:add({ type = "global_max_pool2d" })
model:add({ type = "activation", activation = "relu" })
model:add({ type = "batchnorm" })            -- channels inferred from the stack
model:add({ type = "dropout", rate = 0.2 })  -- train-only; inference is identity
model:add({ type = "flatten" })       -- optional, no parameters
model:add({ units = 1 })
model:compile({ loss = "mse", optimizer = { type = "adam", learning_rate = 0.01 } })
local history = model:fit(x, y, { epochs = 10, batch_size = 32 })
local prediction = model:predict(x)
local probabilities = model:predict_proba(x)  -- softmax probabilities
local classes = model:predict_classes(x)      -- int64 argmax labels
local loss, metrics = model:evaluate(x, y)
model:save("model.ckpt")
model:load("model.ckpt")
model:close()
```

- Inputs are batched Lua tables: `input_shape = { features }` expects
  `{{f1, f2}, ...}`. Convolutional models use NHWC-shaped input:
  `input_shape = { height, width, channels }`; Conv1D uses
  `input_shape = { steps, channels }`; add Flatten before Dense.
- `{ type = "conv2d_transpose", filters = n, kernel_size = k }` builds a
  transposed convolution; its output spatial shape is inferred from `stride`
  and `padding`, or set explicitly with
  `output_shape = { height, width, filters }`.
- `{ type = "embedding", input_dim = n, output_dim = d }` builds text/index
  embeddings; an embedding-first model defaults its input dtype to `int32`.
  Set `input_dtype` explicitly when mixing integer inputs with other layers;
  combine with Flatten before Dense layers.
- Supported losses: `"mse"`/`"mean_squared_error"`,
  `"binary_cross_entropy"`/`"bce"`, `"mae"`/`"mean_absolute_error"`,
  `"softmax_cross_entropy"`/`"cce"`,
  `"sparse_softmax_cross_entropy"`/`"sparse_ce"` (integer labels), or a custom
  function
  `function(graph, predictions, targets) ... end`.
- Supported optimizers: `adam`, `adamw`, `sgd`/`gradient_descent`, `rmsprop`,
  `adagrad`, `adadelta`, `ftrl`, `proximal_adagrad`, `centered_rmsprop`, and
  `momentum`.
- `metrics = { "accuracy" }` adds per-epoch class accuracy; targets are one-hot
  (or integer indices for sparse losses) and the output has class columns.
- `metrics = { "binary_accuracy" }` thresholds probability outputs against
  `0`/`1` targets.
- `class_weight = { [0] = 1, [1] = 2 }` weights sparse class losses; keys are
  TensorFlow class indices (0-based).
- `label_smoothing = 0.1` smooths sparse targets before softmax
  cross-entropy.
- `metrics = { "accuracy", "top_3_accuracy" }` adds top-k accuracy for both
  sparse and one-hot labels.
- `sample_weight = { 1, 2, ... }` scales per-example losses for MSE, MAE,
  BCE, CCE, and sparse CE; pass the same option to `evaluate` for weighted
  validation.
- `fit` accepts `shuffle = true` and `validation_split` (fraction of the tail
  held out when `validation_data` is absent); it returns
  `{ loss = { per_epoch, ... } }`; with
  `validation_data = { x, y }` it also returns `val_loss`.
- `callbacks` may contain tables with `on_train_begin(history, model)`,
  `on_epoch_end(history, epoch, model)`, and `on_train_end(history, model)`
  functions. Set `history.stop_training = true` to end training early.
- `count_params()` returns the total number of weights and biases;
  `summary()` renders the layer stack and total;
- `get_config()` returns a serializable architecture/compile table;
  `save_config(path)` and `load_config(path)` use a trusted Lua-literal file;
  `Sequential.from_config(table)` reconstructs a model;
- `get_weights()` returns variable tensors and `set_weights(list)` assigns them;
- `save`/`load` use the portable tensorflow-lua checkpoint format;
  `save_native`/`load_native` use TensorFlow V2 checkpoints for model weights;
  `save_training`/`load_training` also persist optimizer slots and epoch
  metadata so training can resume (small floating-point drift is possible).
- `export(path)` writes the graph as a GraphDef protobuf.

### Inference-only models

`compile({ optimizer = "none" })` builds a Sequential model without optimizer
updates. It supports `predict`/`evaluate`, and is required for architectures
containing inference-only or unsupported-gradient ops. Calling `fit` then
raises a clear error.

### Tensor mini-batches

`model:compile({ gradient_accumulation_steps = 4 })` accumulates gradients
over micro-batches and applies the optimizer every fourth step, including a
final partial group at epoch end.

`model:fit_tensor(x, y, { batch_size = 32, shuffle = true, seed = 7 })` slices
the first dimension without converting tensors to Lua tables. `evaluate_tensor`
accepts the same options. Row selection is byte-level, so the tensor data must
be contiguous (the normal placeholder/feed path is).

## Functional model

`tf.Functional.new` builds a multi-input, multi-output graph from named inputs.
The builder receives the input table and graph and returns an output table plus
an optional variable list:

```lua
local model = tf.Functional.new({
  inputs = {
    { name = "left", shape = { 1 } },
    { name = "right", shape = { 1 } },
  },
  targets = { sum = { shape = { 1 } } },
  build = function(inputs, graph)
    local scale = tf.variable(graph, 0.5, "scale")
    local sum = tf.add(graph, inputs.left, inputs.right, "sum")
    return { sum = tf.mul(graph, sum, scale.read, "output", "float32") },
      { scale }
  end,
  optimizer = { type = "sgd", learning_rate = 0.01 },
  -- optional loss = function(graph, outputs, targets) return scalar end
})
model:fit({ left = ..., right = ... }, { sum = ... }, { epochs = 10 })
local outputs = model:predict({ left = ..., right = ... })
```

Without a custom loss, the default is the summed mean-squared error over every
declared target. `loss_weights` scales each named output's default loss.

`fit` accepts `callbacks` and `validation_data = { inputs, targets }`.
`get_weights`/`set_weights`, `save(prefix)`/`load(prefix)` (native V2
checkpoints), `export(path)` (GraphDef), `summary()`, `count_params()`, and
`close()` mirror the corresponding Sequential conveniences. This remains a
graph-building API, not a Keras layer registry or architecture serializer.

## Learning-rate schedules

`tf.schedules` returns pure Lua functions of the zero-based step:

```lua
local lr = tf.schedules.exponential_decay(0.1, 1000, 0.5)
local cosine = tf.schedules.cosine_decay(0.1, 1000)
local piecewise = tf.schedules.piecewise_constant(0.1, { 500 }, { 0.1, 0.01 })
```

`Sequential:compile({ optimizer = { learning_rate = schedule } })` calls the
schedule once per epoch. Gradient accumulation is not implemented yet.

## Eager execution (experimental)

```lua
local context = tf.eager.Context.new()
local a = context:constant({ { 1, 2 }, { 3, 4 } })
local b = context:constant({ { 10, 20 }, { 30, 40 } })
local sum = context:add(a, b)
print(sum:totable())       -- { { 11, 22 }, { 33, 44 } }
context:close()            -- deferred until live tensors are released
```

`Context:execute(op_type, inputs, { attrs = ... })` runs an arbitrary eager
op. Type-string attributes (`T`, `dtype`, ...), booleans, numbers, and shape
tables are supported. Attribute values may be booleans, numbers (float), dtype strings, shape arrays,
or typed tables such as `{ int = 0 }`, `{ ints = { 0, 2 } }`,
`{ float = 0.5 }`, `{ bool = true }`, `{ type = "float32" }`,
`{ string = "CONSTANT" }`, and `{ shape = { 2, 3 } }`. Multi-output ops accept
`{ num_outputs = N }`; `Context:split` uses it automatically. Convenience
methods: `add`, `sub`, `mul`, `div`, `matmul`, `maximum`, `minimum`, `cast`,
`clip_by_value`, `transpose`, `expand_dims`, `squeeze`, `stack`, `unstack`,
`concat`, `gather`, `boolean_mask`, `top_k`, `in_top_k`, `depth_to_space`,
`space_to_depth`, `space_to_batch_nd`, `batch_to_space_nd`, `gather_nd`,
`scatter_nd`, `tensor_scatter_nd_update`,
`tensor_scatter_nd_add`, `tensor_scatter_nd_sub`, `tensor_scatter_nd_min`,
`tensor_scatter_nd_max`, `unique`, `unique_with_counts`, `sequence_mask`,
`diag`, `diag_part`, `matrix_diag`, `matrix_diag_part`, `matrix_band_part`,
`matrix_set_diag`, `matrix_inverse`, `matrix_determinant`, `matrix_solve`,
`cholesky`, `matrix_triangular_solve`, `qr`, `self_adjoint_eig`, `svd`,
`bincount`, `norm`, `random_uniform`, `random_normal`, `broadcast_to`, `roll`,
`zeros`, `ones`, `slice`, `pad`, `tile`,
`where`, `reverse`, `reverse_sequence`,
`invert_permutation`, `split`,
`one_hot`, `equal`/`not_equal`/`greater`/`greater_equal`/`less`/`less_equal`,
`logical_and`/`logical_or`/`logical_xor`/`logical_not`,
`reduce_sum`/`reduce_mean`/`reduce_max`/`reduce_min`/`reduce_prod`/
`reduce_all`/`reduce_any`/`reduce_logsumexp`/`reduce_variance`/
`reduce_stddev`, `argmax`/`argmin`, `add_n`, `batch_matmul`, `fill`, `range`,
`linspace`, `relu`, `relu6`, `leaky_relu`, `elu`, `softsign`, `selu`,
`sigmoid`, `tanh`, `softmax`,
`log_softmax`, `softplus`, `exp`, `log`, `sqrt`, `floor`, `ceil`, `round`,
`sign`, `rsqrt`, `reciprocal`, `zeros_like`, `ones_like`, `identity`,
`stop_gradient`, `erf`, `erfc`, `is_nan`, `is_inf`, `is_finite`, `abs`,
`square`, and `reshape`.
Reduction methods accept an integer axes table and an optional `keep_dims`
boolean. `pad` accepts `CONSTANT`, `REFLECT`, or `SYMMETRIC`; `split` accepts a
positive split count or a size table and returns an output list. `stack`,
`unstack`, `one_hot`, and comparisons mirror their graph-mode counterparts;
`clip_by_value` accepts numeric bounds. `logical_xor` uses `NotEqual` because
`LogicalXor` is not registered in the supported C libraries.

`ETensor:destroy()` releases a tensor handle immediately; destroying is
idempotent. `Context:close()` marks the context closed and defers its deletion
until every live eager tensor has been released.

## Training examples

`examples/linear_regression.lua` trains `y = 2x + 1` with gradient descent;
`examples/xor.lua` trains a 2-4-1 ReLU/sigmoid MLP with Adam.

## Typed ops

Graph-output arguments can be operation results or `"name:index"` strings.

| function | operation |
| --- | --- |
| `tf.nn.relu(graph, x, name[, dtype])` | Relu |
| `tf.nn.relu6(graph, x, name[, dtype])` | Relu6 |
| `tf.nn.leaky_relu(graph, x, name[, dtype[, alpha]])` | LeakyRelu |
| `tf.nn.elu(graph, x, name[, dtype])` | Elu |
| `tf.nn.softsign(graph, x, name[, dtype])` | Softsign |
| `tf.nn.selu(graph, x, name[, dtype])` | Selu |
| `tf.nn.sigmoid(graph, x, name[, dtype])` | Sigmoid |
| `tf.nn.tanh(graph, x, name[, dtype])` | Tanh |
| `tf.nn.softmax(graph, x, name[, dtype])` | Softmax |
| `tf.nn.log_softmax(graph, x, name[, dtype])` | LogSoftmax |
| `tf.exp`, `tf.log`, `tf.sqrt`, `tf.floor`, `tf.ceil`, `tf.round`, `tf.sign`, `tf.rsqrt`, `tf.reciprocal` | elementwise unary ops |
| `tf.erf`, `tf.erfc` | elementwise error functions |
| `tf.is_nan`, `tf.is_inf`, `tf.is_finite` | boolean numeric predicates |
| `tf.zeros_like(graph, x[, name[, dtype]])` | ZerosLike |
| `tf.ones_like(graph, x[, name[, dtype]])` | OnesLike |
| `tf.identity(graph, x[, name[, dtype]])` | Identity |
| `tf.stop_gradient(graph, x[, name[, dtype]])` | StopGradient |
| `tf.reshape(graph, x, shape, name[, dtype])` | Reshape |
| `tf.transpose(graph, x, perm[, name[, dtype]])` | Transpose |
| `tf.expand_dims(graph, x, axis[, name[, dtype]])` | ExpandDims |
| `tf.squeeze(graph, x[, axis[, name[, dtype]]])` | Squeeze |
| `tf.gather(graph, params, indices[, axis[, name[, dtype[, indices_dtype]]]])` | GatherV2 |
| `tf.boolean_mask(graph, x, mask[, name[, dtype]])` | Where + Reshape + GatherV2 |
| `tf.depth_to_space(graph, x, block_size[, name[, dtype[, data_format]]])` | DepthToSpace |
| `tf.space_to_depth(graph, x, block_size[, name[, dtype[, data_format]]])` | SpaceToDepth |
| `tf.gather_nd(graph, params, indices[, name[, dtype[, indices_dtype]]])` | GatherNd |
| `tf.scatter_nd(graph, indices, updates, shape[, name[, dtype[, indices_dtype]]])` | ScatterNd |
| `tf.tensor_scatter_nd_update(graph, tensor, indices, updates[, name[, dtype[, indices_dtype]]])` | TensorScatterUpdate |
| `tf.tensor_scatter_nd_add(graph, tensor, indices, updates[, name[, dtype[, indices_dtype]]])` | TensorScatterAdd |
| `tf.tensor_scatter_nd_sub(graph, tensor, indices, updates[, name[, dtype[, indices_dtype]]])` | TensorScatterSub |
| `tf.tensor_scatter_nd_min(graph, tensor, indices, updates[, name[, dtype[, indices_dtype]]])` | TensorScatterMin |
| `tf.tensor_scatter_nd_max(graph, tensor, indices, updates[, name[, dtype[, indices_dtype]]])` | TensorScatterMax |
| `tf.sequence_mask(graph, lengths, maxlen[, name[, dtype]])` | Range + Less + Cast |
| `tf.unique(graph, x[, name[, dtype[, out_idx]]])` | Unique (values, indices) |
| `tf.unique_with_counts(graph, x[, name[, dtype[, out_idx]]])` | UniqueWithCounts (values, indices, counts) |
| `tf.diag(graph, x[, name[, dtype]])` | Diag |
| `tf.diag_part(graph, x[, name[, dtype]])` | DiagPart |
| `tf.matrix_diag(graph, x[, name[, dtype]])` | MatrixDiag |
| `tf.matrix_diag_part(graph, x[, name[, dtype]])` | MatrixDiagPart |
| `tf.matrix_band_part(graph, x, num_lower, num_upper[, name[, dtype]])` | MatrixBandPart |
| `tf.matrix_set_diag(graph, x, diagonal[, name[, dtype]])` | MatrixSetDiag |
| `tf.matrix_inverse(graph, x[, name[, dtype[, adjoint]]])` | MatrixInverse |
| `tf.matrix_determinant(graph, x[, name[, dtype]])` | MatrixDeterminant |
| `tf.matrix_solve(graph, matrix, rhs[, name[, dtype[, adjoint]]])` | MatrixSolve |
| `tf.cholesky(graph, x[, name[, dtype]])` | Cholesky |
| `tf.matrix_triangular_solve(graph, matrix, rhs[, lower[, adjoint[, name[, dtype]]]])` | MatrixTriangularSolve |
| `tf.qr(graph, x[, full_matrices[, name[, dtype]]])` | Qr (q, r) |
| `tf.self_adjoint_eig(graph, x[, compute_v[, name[, dtype]]])` | SelfAdjointEigV2 (values, vectors) |
| `tf.svd(graph, x[, compute_uv[, full_matrices[, name[, dtype]]]])` | Svd (s, u, v) |
| `tf.bincount(graph, values, size[, weights[, name[, dtype]]])` | Bincount |
| `tf.random_uniform(graph, shape[, name[, dtype[, seed]]])` | RandomUniform |
| `tf.random_normal(graph, shape[, name[, dtype[, seed]]])` | RandomStandardNormal |
| `tf.norm(graph, x, axes[, ord[, name[, dtype[, keep_dims]]]])` | composed L1/L2 norm |
| `tf.broadcast_to(graph, x, shape[, name[, dtype]])` | BroadcastTo |
| `tf.roll(graph, x, shift, axis[, name[, dtype]])` | Roll |
| `tf.reduce_mean(graph, x, axes[, name[, dtype[, keep_dims]]])` | Mean |
| `tf.reduce_sum`, `tf.reduce_max`, `tf.reduce_min` | Sum/Max/Min reductions |
| `tf.reduce_prod(graph, x, axes[, name[, dtype[, keep_dims]]])` | Prod |
| `tf.reduce_all`, `tf.reduce_any` | All/Any boolean reductions |
| `tf.reduce_variance(graph, x, axes[, name[, dtype[, keep_dims]]])` | composed variance |
| `tf.reduce_stddev(graph, x, axes[, name[, dtype[, keep_dims]]])` | composed standard deviation |
| `tf.equal`, `tf.not_equal`, `tf.greater`, `tf.greater_equal`, `tf.less`, `tf.less_equal` | elementwise comparisons returning bool |
| `tf.logical_and`, `tf.logical_or`, `tf.logical_xor`, `tf.logical_not` | boolean logic |
| `tf.squared_difference(graph, a, b[, name[, dtype]])` | SquaredDifference |
| `tf.add_n(graph, inputs[, name[, dtype]])` | AddN |
| `tf.batch_matmul(graph, a, b[, name[, dtype[, opts]]])` | BatchMatMulV2 |
| `tf.fill(graph, dims, value[, name[, dtype]])` | Fill |
| `tf.zeros(graph, shape[, name[, dtype]])` | Fill(0) |
| `tf.ones(graph, shape[, name[, dtype]])` | Fill(1) |
| `tf.range(graph, start, limit, delta[, name[, dtype]])` | Range |
| `tf.linspace(graph, start, stop, num[, name[, dtype]])` | LinSpace |
| `tf.reduce_logsumexp(graph, x, axes[, name[, dtype[, keep_dims]]])` | stable composed log-sum-exp |
| `tf.argmax(graph, x, axis[, name[, dtype[, output_dtype]]])` | ArgMax |
| `tf.argmin(graph, x, axis[, name[, dtype[, output_dtype]]])` | ArgMin |
| `tf.top_k(graph, x, k[, name[, dtype[, sorted]]])` | TopKV2 (values, indices) |
| `tf.in_top_k(graph, predictions, targets, k[, name[, targets_dtype]])` | InTopK |
| `tf.space_to_batch_nd(graph, x, block_shape, paddings[, name[, dtype]])` | SpaceToBatchND |
| `tf.batch_to_space_nd(graph, x, block_shape, crops[, name[, dtype]])` | BatchToSpaceND |
| `tf.one_hot(graph, indices, depth[, opts])` | OneHot |
| `tf.cast(graph, x, src_dtype, dst_dtype, name)` | Cast |
| `tf.clip_by_value(graph, x, min, max[, name[, dtype]])` | ClipByValue |
| `tf.shape(graph, x[, name[, dtype]])` | Shape |
| `tf.div(graph, a, b[, name[, dtype]])` | RealDiv |
| `tf.concat(graph, {a, b}, axis[, name[, dtype]])` | ConcatV2 |
| `tf.split(graph, x, count_or_sizes[, axis[, name[, dtype]]])` | Split/SplitV |
| `tf.pad(graph, x, paddings[, mode[, name[, dtype]]])` | Pad/MirrorPad |
| `tf.tile(graph, x, multiples[, name[, dtype]])` | Tile |
| `tf.where(graph, condition, x, y[, name[, dtype]])` | SelectV2 |
| `tf.stack(graph, inputs[, axis[, name[, dtype]]])` | Pack |
| `tf.unstack(graph, x, num[, axis[, name[, dtype]]])` | Unpack |
| `tf.slice(graph, x, begin, size[, name[, dtype]])` | Slice |
| `tf.reverse(graph, x, axis[, name[, dtype]])` | ReverseV2 |
| `tf.reverse_sequence(graph, x, lengths[, seq_dim[, batch_dim[, name[, dtype[, lengths_dtype]]]]])` | ReverseSequence |
| `tf.invert_permutation(graph, x[, name[, dtype]])` | InvertPermutation |
| `tf.nn.softplus(graph, x, name[, dtype])` | Softplus |

`tf.split` and `tf.unstack` return lists of outputs; a numeric split count uses
`Split` and a size table uses `SplitV`. `tf.unique` returns
`{ values, indices }`; `tf.unique_with_counts` returns
`{ values, indices, counts }`. Their `out_idx` defaults to `int32` and applies
to the index/count outputs. `tf.boolean_mask` expects a rank-1 boolean mask and
gathers matching slices along axis 0; `tf.depth_to_space` and
`tf.space_to_depth` accept `NHWC` or `NCHW` and a positive `block_size`.
`tf.sequence_mask` builds a `[batch, maxlen]` mask with a positive integer
`maxlen` and casts the result to `dtype` (default `float32`). `tf.top_k` returns
`{ values, indices }`; `tf.in_top_k` expects rank-2 prediction scores and a
rank-1 target vector. `tf.space_to_batch_nd` and `tf.batch_to_space_nd` validate
that the padding/crop rows match the block shape rank. `tf.pad` accepts
`CONSTANT` (zero
padding), `REFLECT`, or `SYMMETRIC`; `tf.where` uses broadcasting `SelectV2`.
`tf.slice` accepts matching `begin` and `size` tables where a size of `-1`
selects all remaining elements in that dimension. `tf.reverse` accepts an
integer axis or a table of axes. `tf.add_n` sums a non-empty output list.
`tf.batch_matmul` accepts `opts.adj_x`/`opts.adj_y` (aliases
`transpose_a`/`transpose_b`). `tf.fill` requires non-negative integer
dimensions; `tf.range` follows TensorFlow's scalar start/limit/delta rules;
`tf.linspace` requires a positive integer `num` and a float32/float64 dtype.
`tf.reduce_logsumexp` subtracts the per-axis maximum before exponentiating, so
it stays finite for large inputs, and is composed of differentiable ops rather
than a fused kernel. `tf.reduce_variance` and `tf.reduce_stddev` are composed
from `Mean`/`SquaredDifference`/`Sqrt`, support `keep_dims`, and infer their
dtype from the input unless one is supplied. `tf.zeros_like`/`ones_like` keep
the input shape and dtype, and `tf.stop_gradient` maps to `StopGradient`.
`tf.erf`/`erfc` preserve the input floating dtype, while `tf.is_nan`,
`tf.is_inf`, and `tf.is_finite` return `bool` tensors. `tf.argmax`/`tf.argmin`
validate that the axis is an integer. `tf.logical_xor` is built from `NotEqual`
because the `LogicalXor` op is not registered in the supported C libraries;
comparison and logical ops return `bool` outputs.

### `tf.gradients(graph, outputs, inputs[, grad_ys])`

Adds TensorFlow gradient operations for a list of output graph outputs with
respect to a list of input outputs. Returns one graph output per input. When
`grad_ys` is supplied, it must contain one upstream gradient output per entry
in `outputs`; otherwise TensorFlow uses an `OnesLike` seed for each output.
This wraps `TF_AddGradients`; TensorFlow's C API gradient coverage is more
limited than Python's.

### `graph:operation(name)`

Returns the raw operation handle; raises a Lua error if absent.

### `graph:graph_def()`

Serializes the graph to bytes.

### `graph:delete()`

Deletes the TF_Graph immediately. Any live session must be closed first; raw
operation handles become invalid.

## Deployment bundles

A portable bundle contains `graph.pb`, a native V2 checkpoint at `variables`,
and a `manifest.lua`:

```lua
model:save_bundle("/tmp/model.bundle")
local bundle = tf.load_bundle("/tmp/model.bundle")
local output = bundle.session:run({
  [bundle.inputs[1]] = input_tensor,
}, { bundle.outputs[1] })[1]
```

`tf.save_bundle(directory, graph, session, variables, metadata)` is the
low-level form. `tf.load_bundle` returns `{ graph, session, manifest, inputs,
outputs }`. This is not official SavedModel export; the C API does not expose
the MetaGraphDef writer.

## Sessions

### `tf.Session.new(graph[, opts])`

Creates a session. `opts.target` sets the TensorFlow target string.

### `session:run(feeds, fetches[, targets])`

- `feeds` maps operation names to `tf.Tensor` values or Lua tables. When the
  feed key is a placeholder graph output, its dtype is used for table
  conversion.
- `fetches` is a `"op:index"` string, a list of such strings, or a named map
  `{ result = "op:index" }`.
- `targets` is a single operation name or a list.

A single string fetch returns one Tensor; a list returns an array; a named map
returns a table keyed by the names. Feed tensors remain owned by Lua.

```lua
local z = session:run({ x = { 1, 2 }, y = { 3, 4 } }, "z:0")
local named = session:run({ x = { 1 }, y = { 2 } }, { sum = "z:0" })
```

### `session:close()`

Closes and deletes the underlying TF_Session. Idempotent, and safe to call
explicitly. An `ffi.gc` finalizer is a fallback; explicit close is preferred.
The session keeps its graph handle alive.

### `tf.load_saved_model(path[, tags[, opts]])`

Returns a `Session`, `Graph`, and decoded `signatures` table loaded from a
SavedModel export directory. Default tags are `{ "serve" }`.

```lua
local session, graph, signatures = tf.load_saved_model("model")
-- signatures.serving_default.inputs.x.name == "serving_default_x:0"
local outputs = session:run_signature("serving_default",
  { x = { 1, 2 }, y = { 10, 20 } })
```

### `session:run_signature(name, feeds)`

Maps signature input names to operation names, runs the graph, and returns a
table keyed by the signature output names. Feeds may be `tf.Tensor` objects or
rectangular Lua tables. The SavedModel initializer is run by TensorFlow during
session loading.

## Errors

All C API calls check `TF_Status`. Failures raise Lua errors with TensorFlow's
message; malformed tensors and bad operation names are rejected before the C
call where possible.
