# tensorflow-lua API

## Loading

```lua
local tf = require("tensorflow")
print(tf.version())          -- "2.18.0"
```

The module loads `libtensorflow.so` from `TENSORFLOW_LIB`, `TENSORFLOW_ROOT`,
`third_party/libtensorflow`, or the dynamic linker path.

`tf.VERSION` is the binding version; `tf.version()` returns libtensorflow's
version.

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

## Graphs

### `tf.Graph.new()`

Creates a TF_Graph.

### `graph:import_graph_def(bytes)`

Imports a serialized `GraphDef` protobuf from a Lua string.

### `graph:import_file(path)`

Reads and imports a `.pb` file.

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

### `tf.train.gradient_descent(graph, loss, variables, lr[, name])`

Adds `ApplyGradientDescent` targets for the variables and returns
`targets, gradients, learning_rate`. Initialize variables first with
`session:run({}, {}, { variable.initializer })`, then run the target list once
per training step.

### `tf.train.adam(graph, loss, variables[, opts])`

Adds `ApplyAdam` updates and slot variables. Options: `learning_rate`/`lr`
(default `0.001`), `beta1` (0.9), `beta2` (0.999), `epsilon` (1e-8), `name`.
Options also include `weight_decay` and `clipvalue` (gradient-value clamping).
Returns `targets, gradients, state`.
Initialize each variable and every entry in `state.initializers` before
training.

### `tf.train.adamw(graph, loss, variables[, opts])`

Adam with decoupled weight decay; defaults `weight_decay = 0.01`. Accepts the
same options and returns the same triple as `tf.train.adam`.

### `tf.train.rmsprop(graph, loss, variables[, opts])`

Adds `ApplyRMSProp` updates and slot variables. Options: `learning_rate`/`lr`
(default `0.001`), `rho` (0.9), `momentum` (0), `epsilon` (1e-10), `name`.
Returns `targets, gradients, state`; initialize `state.initializers`.

### `tf.train.Saver.new(graph, variables)`

Reusable native V2 checkpoint saver. `variables` is a list of
`{ name = ..., variable = ... }` entries. Methods: `save(session, prefix)`,
`restore(session, prefix)`, and `names()`.

### `tf.train.centered_rmsprop(graph, loss, variables[, opts])`

Adds `ApplyCenteredRMSProp` updates with mean-gradient, variance, and momentum
slots. Options: `learning_rate`/`lr` (default `0.001`), `rho` (0.9),
`momentum` (0), `epsilon` (1e-10), `name`. Initialize `state.initializers`.

### `tf.train.adagrad(graph, loss, variables[, opts])`

Adds `ApplyAdagrad` updates and accumulator variables. Options:
`learning_rate`/`lr` (default `0.01`), `name`. Initialize
`state.initializers`.

### `tf.train.adadelta(graph, loss, variables[, opts])`

Adds `ApplyAdadelta` updates with accumulator and update slots. Options:
`learning_rate`/`lr` (default `1.0`), `rho` (0.95), `epsilon` (1e-8), `name`.
Initialize `state.initializers`.

### `tf.train.proximal_adagrad(graph, loss, variables[, opts])`

Adds FOBOS/Adagrad updates with an accumulator slot. Options:
`learning_rate`/`lr` (default `0.01`), `l1` (0), `l2` (0), `name`. Initialize
`state.initializers`.

### `tf.train.ftrl(graph, loss, variables[, opts])`

Adds Ftrl-Proximal updates with accumulator and linear slots. Options:
`learning_rate`/`lr` (default `0.001`), `l1` (0), `l2` (0), `lr_power`
(`-0.5`), `name`. Initialize `state.initializers`.

### `tf.train.momentum(graph, loss, variables[, opts])`

Adds `ApplyMomentum` updates and accumulator variables. Options:
`learning_rate`/`lr` (default `0.01`), `momentum` (0.9), `name`. Initialize
`state.initializers`.

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
`tf.save_checkpoint`. Returns the target list for `session:run`.

### `tf.CheckpointReader`

Reads official TensorFlow checkpoint files:

```lua
local reader = tf.CheckpointReader.new("model.ckpt")
print(reader:size())
for _, name in ipairs(reader:variables()) do print(name) end
local tensor = reader:tensor(name)
reader:close()
```

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

## Data batching

```lua
local dataset = tf.data.Dataset.new(x, y):shuffle(7):batch(32)
for batch in dataset:each() do
  -- batch.x and batch.y
end
```

`Dataset.new(x[, y])`, `size()`, `batch(size)`, `shuffle([seed])`,
`split(fraction)`, `repeat_times(count)`, `take(count)`, and `each()` return
batches as `{ x = ..., y = ... }`. (`repeat` is a Lua keyword, so the method is
`repeat_times`.)

`model:fit(dataset, opts)` trains directly from labeled dataset batches;
`fit_dataset(dataset, opts)` is an alias. `Dataset:split(fraction)` returns
train/validation datasets, and `model:evaluate_dataset` /
`model:predict_dataset` accept Dataset objects directly.

## Callbacks

### `tf.callbacks.EarlyStopping(opts)`

Stops when a monitored history entry stops improving. Options: `monitor`
(default `val_loss`), `mode` (`min`/`max`), `patience`, `min_delta`, and
`restore_best_weights`. Records `stopped_epoch`; restoration uses the portable
checkpoint format.

### `tf.callbacks.ModelCheckpoint(path, opts)`

Saves the model after epochs. `path` may contain `{epoch}`. Options: `monitor`,
`mode`, `save_best_only`, `min_delta`.

## Metrics

### `tf.metrics.binary_accuracy(graph, predictions, labels[, opts])`

Thresholds probability predictions (default `0.5`) and compares them with
`0`/`1` labels. Options: `threshold`, `dtype`, `labels_dtype`, `axes`, `name`.

### `tf.metrics.accuracy(graph, predictions, labels[, opts])`

Compares integer class predictions to labels with `Equal`, casts to float32,
and reduces the mean over the batch. Options: `dtype` (default `int32`), `axes`,
`name`.

## Layers

### `tf.layers.dense(graph, input, units, opts)`

Builds a Dense layer with a `[input_size, units]` weight variable and optional
bias. Required option: `input_size`. Other options: `use_bias` (true),
`activation` (`"relu"`, `"sigmoid"`, `"tanh"`, a custom function, or nil),
`dtype`, `name`.

Returns `{ output, weight, bias, variables, initializers, input_size, units }`.

### `tf.layers.dropout(graph, input, rate[, opts])`

Builds an inverted-dropout mask from the dynamic input shape. Options:
`training` (default true; false returns the input unchanged), `seed`, `seed2`,
`dtype`, `name`. Returns `{ output }`.

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
model:add({ units = 1 })
model:compile({ loss = "mse", optimizer = { type = "adam", learning_rate = 0.01 } })
local history = model:fit(x, y, { epochs = 10, batch_size = 32 })
local prediction = model:predict(x)
local classes = model:predict_classes(x)  -- int64 argmax labels
local loss, metrics = model:evaluate(x, y)
model:save("model.ckpt")
model:load("model.ckpt")
model:close()
```

- Inputs are batched Lua tables: `input_shape = { features }` expects
  `{{f1, f2}, ...}`.
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
- `fit` accepts `shuffle = true` and `validation_split` (fraction of the tail
  held out when `validation_data` is absent); it returns
  `{ loss = { per_epoch, ... } }`; with
  `validation_data = { x, y }` it also returns `val_loss`.
- `callbacks` may contain tables with `on_train_begin(history, model)`,
  `on_epoch_end(history, epoch, model)`, and `on_train_end(history, model)`
  functions. Set `history.stop_training = true` to end training early.
- `count_params()` returns the total number of weights and biases;
  `summary()` renders the layer stack and total;
- `save`/`load` use the portable tensorflow-lua checkpoint format;
  `save_native`/`load_native` use TensorFlow V2 checkpoints.
- `export(path)` writes the graph as a GraphDef protobuf.

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
or typed tables such as `{ int = 0 }`, `{ float = 0.5 }`,
`{ bool = true }`, `{ type = "float32" }`, and `{ shape = { 2, 3 } }`.
Convenience methods: `add`, `sub`, `mul`, `matmul`, `relu`, `sigmoid`,
`square`, and `reshape`.

## Training examples

`examples/linear_regression.lua` trains `y = 2x + 1` with gradient descent;
`examples/xor.lua` trains a 2-4-1 ReLU/sigmoid MLP with Adam.

## Typed ops

Graph-output arguments can be operation results or `"name:index"` strings.

| function | operation |
| --- | --- |
| `tf.nn.relu(graph, x, name[, dtype])` | Relu |
| `tf.nn.leaky_relu(graph, x, name[, dtype[, alpha]])` | LeakyRelu |
| `tf.nn.elu(graph, x, name[, dtype])` | Elu |
| `tf.nn.sigmoid(graph, x, name[, dtype])` | Sigmoid |
| `tf.nn.tanh(graph, x, name[, dtype])` | Tanh |
| `tf.nn.softmax(graph, x, name[, dtype])` | Softmax |
| `tf.exp`, `tf.log`, `tf.sqrt` | elementwise unary ops |
| `tf.reshape(graph, x, shape, name[, dtype])` | Reshape |
| `tf.reduce_mean(graph, x, axes[, name[, dtype[, keep_dims]]])` | Mean |
| `tf.reduce_sum`, `tf.reduce_max`, `tf.reduce_min` | Sum/Max/Min reductions |
| `tf.argmax(graph, x, axis[, name[, dtype[, output_dtype]]])` | ArgMax |
| `tf.one_hot(graph, indices, depth[, opts])` | OneHot |
| `tf.cast(graph, x, src_dtype, dst_dtype, name)` | Cast |
| `tf.clip_by_value(graph, x, min, max[, name[, dtype]])` | ClipByValue |
| `tf.shape(graph, x[, name[, dtype]])` | Shape |
| `tf.div(graph, a, b[, name[, dtype]])` | RealDiv |
| `tf.concat(graph, {a, b}, axis[, name[, dtype]])` | ConcatV2 |
| `tf.nn.softplus(graph, x, name[, dtype])` | Softplus |

### `tf.gradients(graph, outputs, inputs)`

Adds TensorFlow gradient operations for a list of output graph outputs with
respect to a list of input outputs. Returns one graph output per input. This
wraps `TF_AddGradients`; TensorFlow's C API gradient coverage is more limited
than Python's.

### `graph:operation(name)`

Returns the raw operation handle; raises a Lua error if absent.

### `graph:graph_def()`

Serializes the graph to bytes.

### `graph:delete()`

Deletes the TF_Graph immediately. Any live session must be closed first; raw
operation handles become invalid.

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
