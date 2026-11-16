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
TF_Tensor backing. `opts.dtype` defaults to `float32`. Numeric dtypes are converted explicitly; no
implicit dtype promotion occurs. `string` tensors accept Lua strings and use
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
Returns `targets, gradients, state`. Initialize each variable and every entry in
`state.initializers` before training.

### `tf.train.rmsprop(graph, loss, variables[, opts])`

Adds `ApplyRMSProp` updates and slot variables. Options: `learning_rate`/`lr`
(default `0.001`), `rho` (0.9), `momentum` (0), `epsilon` (1e-10), `name`.
Returns `targets, gradients, state`; initialize `state.initializers`.

### `tf.train.adagrad(graph, loss, variables[, opts])`

Adds `ApplyAdagrad` updates and accumulator variables. Options:
`learning_rate`/`lr` (default `0.01`), `name`. Initialize
`state.initializers`.

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

### `tf.losses.softmax_cross_entropy(graph, logits, labels[, opts])`

Builds a `SoftmaxCrossEntropyWithLogits` mean over the batch axis (default
`axes = {0}`). Options: `axes`, `keep_dims`, `dtype`, `name`.

## Metrics

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

### `tf.layers.sequential(graph, input, definitions, input_size)`

Chains Dense layers. Each definition is `{ units = n, options = {...} }`.
Returns the layer list. The caller must run every layer initializer.

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
| `tf.nn.sigmoid(graph, x, name[, dtype])` | Sigmoid |
| `tf.nn.tanh(graph, x, name[, dtype])` | Tanh |
| `tf.nn.softmax(graph, x, name[, dtype])` | Softmax |
| `tf.exp`, `tf.log`, `tf.sqrt` | elementwise unary ops |
| `tf.reshape(graph, x, shape, name[, dtype])` | Reshape |
| `tf.reduce_mean(graph, x, axes[, name[, dtype[, keep_dims]]])` | Mean |
| `tf.argmax(graph, x, axis[, name[, dtype[, output_dtype]]])` | ArgMax |
| `tf.one_hot(graph, indices, depth[, opts])` | OneHot |

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

- `feeds` maps operation names to `tf.Tensor` values or Lua tables.
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
