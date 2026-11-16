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
TF_Tensor backing. `opts.dtype` defaults to `float32`. Numeric dtypes are
converted explicitly; no implicit dtype promotion occurs.

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
| `t:handle()` | raw `TF_Tensor*` (advanced) |
| `tostring(t)` | shape/dtype summary |

Tensors own their output buffers and delete them in `__gc`.

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

Create `AddV2` and `MatMul` operations with the given input outputs. `dtype`
defaults to `float32`.

### `tf.gradients(graph, outputs, inputs)`

Adds TensorFlow gradient operations for a list of output graph outputs with
respect to a list of input outputs. Returns one graph output per input. This
wraps `TF_AddGradients`; TensorFlow's C API gradient coverage is more limited
than Python's.

### `graph:operation(name)`

Returns the raw operation handle; raises a Lua error if absent.

### `graph:graph_def()`

Serializes the graph to bytes.

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

Closes and deletes the underlying TF_Session. Idempotent.

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
