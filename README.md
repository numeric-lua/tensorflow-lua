# tensorflow-lua

Native LuaJIT bindings for the **TensorFlow C API**. It runs TensorFlow graphs
in-process; it is not a Python wrapper and does not use a second tensor engine.

## Status

`0.1.0` alpha. First milestone: **graph inference**.

- load serialized `GraphDef` files;
- create and inspect dense tensors;
- run sessions with feeds, fetches, and target ops;
- optional SavedModel session loading (experimental; operation names are not
  decoded from signatures yet);
- clear TensorFlow status errors and LuaJIT `__gc` cleanup.

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

Tensors can be created explicitly with `tf.tensor(value, { dtype = "float32" })`.
Supported dtypes are `float32`, `float64`, `int32`, `int64`, `int16`, `int8`,
`uint8`, `uint16`, `uint32`, `uint64`, and `bool`.

## API

See [docs/api.md](docs/api.md).

## License

BSD-3-Clause. libtensorflow is downloaded separately under the Apache 2.0
license.
