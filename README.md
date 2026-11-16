# tensorflow-lua

Native LuaJIT bindings for the **TensorFlow C API**. It runs TensorFlow graphs
in-process; it is not a Python wrapper and does not use a second tensor engine.

## Status

`0.2.0` alpha. Graph inference, graph construction, training, and checkpoints.

- load serialized `GraphDef` files;
- build graphs from Lua with operations, inputs, and attributes;
- add symbolic gradients, variables, gradient-descent updates, and Adam;
- train with gradient descent, Adam, or RMSProp;
- write native TensorFlow V2 checkpoints (`SaveV2`/`RestoreV2`);
- save/restore tensorflow-lua checkpoints and read official TensorFlow
  checkpoints;
- typed ops for reshape, reductions, argmax, and common activations;
- create and inspect dense tensors;
- run sessions with feeds, fetches, and target ops;
- SavedModel loading with signature decoding and named signature execution;
- clear TensorFlow status errors, `ffi.gc`-backed ownership, and explicit
  `Tensor:destroy` / `Graph:delete` / `Session:close` paths.

Not yet implemented: a Lua graph builder, TensorFlow eager execution, string/
variant tensors, checkpoints, and training optimizer helpers.

## Requirements

- LuaJIT 2.1
- Linux x86_64 libtensorflow 2.18 (download it with
  `./scripts/fetch_libtensorflow.sh`)
- no Lua rocks

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh

# optional LuaRocks install (pure Lua, external libtensorflow)
luarocks --lua-version=5.1 make tensorflow-lua-scm-1.rockspec
```

Set `TENSORFLOW_ROOT` to the extracted libtensorflow directory or
`TENSORFLOW_LIB` to the shared object if it lives elsewhere.

## Quick start

Export a GraphDef from Python:

```python
import tensorflow as tf

with tf.Graph().as_default():
    x = tf.compat.v1.placeholder(tf.float32, [None], name="x")
    y = tf.compat.v1.placeholder(tf.float32, [None], name="y")
    z = tf.add(x, y, name="z")
    data = tf.compat.v1.GraphDef()
    data.ParseFromString(tf.compat.v1.get_default_graph().as_graph_def()
                         .SerializeToString())
    open("add.pb", "wb").write(data.SerializeToString())
```

Run it from Lua:

```lua
local tf = require("tensorflow")

local graph = tf.Graph.new():import_file("add.pb")
local session = tf.Session.new(graph)
local result = session:run(
  { x = { 1, 2, 3 }, y = { 10, 20, 30 } },
  { "z:0" })
print(result[1]:totable())   -- { 11, 22, 33 }
session:close()
```

SavedModels are first-class:

```lua
local session, graph, signatures = tf.load_saved_model("my_saved_model")
local outputs = session:run_signature("serving_default",
  { x = { 1, 2 }, y = { 10, 20 } })
print(outputs.sum:totable())
```

Graphs can also be built from Lua:

```lua
local graph = tf.Graph.new()
local x = tf.placeholder(graph, "float32", { 2 }, "x")
local y = tf.placeholder(graph, "float32", { 2 }, "y")
local z = tf.add(graph, x, y, "z")
local session = tf.Session.new(graph)
local result = session:run({ [x] = { 1, 2 }, [y] = { 10, 20 } }, { z })
print(result[1]:totable())   -- { 11, 22 }
```

Variables and graph-mode training are available:

```lua
local variable = tf.variable(graph, 5, "v")
local square = graph:new_operation("Square", "loss")
square:add_input(variable.read)
square:set_type("T", "float32")
local loss = square:finish()
local updates = tf.train.gradient_descent(graph, loss, { variable }, 0.1)

local session = tf.Session.new(graph)
session:run({}, {}, { variable.initializer })
for _ = 1, 50 do session:run({}, {}, updates) end
print(session:run({}, { variable.read })[1]:totable())
```

Adam is available with slot variables and bias-correction state:

```lua
local updates, _, state = tf.train.adam(graph, loss, { variable },
  { learning_rate = 0.01 })
session:run({}, {}, { variable.initializer, state.initializers[1],
  state.initializers[2], state.initializers[3] })
```

Tensor checkpoints can be saved and restored:

```lua
tf.save_checkpoint(session, "model.ckpt", { { name = "v", variable = v } })
local restore = tf.load_checkpoint(graph2, "model.ckpt", { { name = "v", variable = v2 } })
```

Native TensorFlow V2 checkpoints are written directly from a graph:

```lua
local save = tf.save_v2(graph, "/tmp/model/ckpt", { { name = "v", variable = v } })
session:run({}, {}, { save })
local restore = tf.restore_v2(graph2, "/tmp/model/ckpt",
  { { name = "v", variable = v2 } })
```

`tf.CheckpointReader` reads official TensorFlow checkpoint files.

Symbolic gradients are also directly available:

```lua
local square = graph:new_operation("Square", "square")
square:add_input(x)
square:set_type("T", "float32")
local y = square:finish()
local dx = tf.gradients(graph, { y }, { x })[1]
```

Tensors can be created explicitly with `tf.tensor(value, { dtype = "float32" })`.
Supported dtypes are `float32`, `float64`, `int32`, `int64`, `int16`, `int8`,
`uint8`, `uint16`, `uint32`, `uint64`, `bool`, and `string`. String tensors use
TensorFlow's native `TF_TString` representation.

`tf.VERSION` is the binding version; `tf.version()` returns the loaded
libtensorflow version.

## Examples

```sh
TENSORFLOW_ROOT=$PWD/third_party/libtensorflow \
  luajit examples/linear_regression.lua
TENSORFLOW_ROOT=$PWD/third_party/libtensorflow \
  luajit examples/xor.lua
```

The examples build and train linear regression and a 2-4-1 XOR MLP entirely
from Lua. The XOR example uses Adam.

## API

See [docs/api.md](docs/api.md).

## License

BSD-3-Clause. libtensorflow is downloaded separately under the Apache 2.0
license.
