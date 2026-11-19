# tensorflow-lua

Native LuaJIT bindings for the **TensorFlow C API**. It runs TensorFlow graphs
in-process; it is not a Python wrapper and does not use a second tensor engine.

[![❤ Sponsor on GitHub](https://img.shields.io/badge/%E2%9D%A4%20Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Mateus-M-Reis)

## Status

`0.17.0` alpha. Graph inference, construction, training, checkpoints, eager
mode, trainable 2-D/3-D convolution and pooling, attention/transformer layers,
a checkpointable functional graph model, and a Sequential model. The
dependency-free suite currently contains 170 tests.

- load serialized `GraphDef` files;
- build graphs from Lua with operations, inputs, and attributes;
- add symbolic gradients, variables, gradient-descent updates, and Adam;
- train with gradient descent, Adam/AdamW, RMSProp/CenteredRMSProp, Adagrad,
  Adadelta, Ftrl, ProximalAdagrad, or momentum SGD, with global-norm gradient
  clipping;
- write native TensorFlow V2 checkpoints (`SaveV2`/`RestoreV2`);
- save/restore tensorflow-lua checkpoints and read official TensorFlow
  checkpoints;
- typed ops for reshape, reductions, argmax, and common activations;
- experimental eager execution (`tf.eager.Context`);
- compositional Dense, Conv1D, Conv2D, Conv2DTranspose, depthwise and
  separable Conv2D, 1-D/2-D pooling, BatchNormalization, LayerNormalization,
  Activation, Reshape, Permute, UpSampling2D, ZeroPadding2D, Embedding,
  Flatten, and Dropout layers, including pooling stacks, resumable BatchNorm
  statistics, and runtime train/eval normalization/dropout switching;
- tested against libtensorflow 2.16.2 and 2.18.0 on Ubuntu 22.04/24.04;
- loss helpers (MSE, MAE, Huber, hinge, categorical hinge,
  binary/softmax/sparse-softmax cross-entropy), binary/class accuracy,
  precision/recall/F1, cast, concat, and OneHot;
- a minimal `tf.Sequential` model with fit/evaluate/predict/save/load,
  validation loss, ReduceLROnPlateau/EarlyStopping/ModelCheckpoint callbacks,
  accuracy and top-k metrics,
  sample-weighted dense and sparse losses, class-weighted and label-smoothed
  sparse losses,
  `predict_proba`/`predict_classes`, weight snapshots, Embedding/Flatten
  stacks, tensor-backed mini-batch `fit_tensor`, runtime train/eval Dropout,
  resumable optimizer/epoch state, architecture config save/load, GraphDef
  export, and portable/native checkpoints;
- create and inspect dense tensors;
- run sessions with feeds, fetches, and target ops;
- SavedModel loading with signature decoding and named signature execution;
- clear TensorFlow status errors, `ffi.gc`-backed ownership, and explicit
  `Tensor:destroy` / `Graph:delete` / `Session:close` paths.

- `tf.Functional` for multi-input, multi-output graphs built from named
  placeholders and arbitrary TensorFlow operations;
- pure Lua exponential, cosine, polynomial, piecewise, and constant
  learning-rate schedules;
- graph-level gradient accumulation for Sequential training;
- binary AUC and macro multiclass precision/recall/F1;
- `tf.Functional` checkpointing, weight get/set, GraphDef export, summaries,
  callbacks, validation loss, and per-output loss weights;
- trainable Conv3D built from depth slices plus portable Conv2D kernels, with
  SAME/VALID padding, strides, and dilation;
- MaxPool3D, AvgPool3D, GlobalAvgPool3D, and GlobalMaxPool3D layers;
- scaled dot-product attention and multi-head attention with optional causal
  masking;
- sinusoidal positional encoding and a pre-norm transformer block;
- `optimizer = "none"` inference-only Sequential models.

Not yet implemented: official SavedModel export, GPU placement, complete
Keras-style functional layers, and verified libtensorflow 2.19/2.20 support.
Eager execution is available as an experimental `tf.eager` surface with basic
arithmetic and activations.

## Requirements

- LuaJIT 2.1
- Linux x86_64 libtensorflow 2.16.2 or 2.18.0 (download it with
  `./scripts/fetch_libtensorflow.sh`; set `TENSORFLOW_VERSION` to select)
- no mandatory Lua rocks (the rockspec is optional packaging)

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh

# optional LuaRocks install from the public registry
luarocks --lua-version=5.1 install tensorflow-lua

# or install the release source rock directly
luarocks --lua-version=5.1 install \
  https://github.com/numeric-lua/tensorflow-lua/releases/download/v0.17.0/tensorflow-lua-0.17.0-1.src.rock

# or build from a checkout
luarocks --lua-version=5.1 make tensorflow-lua-0.17.0-1.rockspec
./scripts/package.sh tensorflow-lua-0.17.0-1.rockspec
```

Set `TENSORFLOW_ROOT` to the extracted libtensorflow directory or
`TENSORFLOW_LIB` to the shared object if it lives elsewhere. The 2.18.0
Linux x86-64 archive checksum is pinned in the fetch script; override
`TENSORFLOW_SHA256` to verify a different build.

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
Without a dtype, strings, booleans, and complex values infer `string`, `bool`,
and `complex64`; numeric tables default to `float32`.
Supported dtypes are `float32`, `float64`, `int32`, `int64`, `int16`, `int8`,
`uint8`, `uint16`, `uint32`, `uint64`, `bool`, `string`, and quantized
`qint8`, `quint8`, `qint16`, `quint16`, `qint32`, `complex64`, `complex128`,
`float16`, and `bfloat16`. String tensors use TensorFlow's native `TF_TString`
representation. Complex values are built with `tf.complex(real, imag)`.

`tf.VERSION` is the binding version; `tf.version()` returns the loaded
libtensorflow version.

## Sequential model

```lua
local model = tf.Sequential.new({ input_shape = { 1 } })
model:add({ units = 1 })
model:compile({ optimizer = { type = "adam", learning_rate = 0.1 },
                loss = "mse" })
model:fit({ { 0 }, { 1 }, { 2 }, { 3 } }, { { 1 }, { 3 }, { 5 }, { 7 } },
  { epochs = 200, batch_size = 4 })
print(model:predict({ { 4 } }):totable()[1][1])
model:save("linear.ckpt")
model:close()
```

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

## Benchmark

```sh
./scripts/bench.sh
```

See [docs/performance.md](docs/performance.md) for the recorded baseline and
caveats.

The harness measures tensor conversion, graph session execution, Sequential
inference, and training epochs; it is not a CI gate.

## Design and contributing

See [docs/design.md](docs/design.md) for FFI ownership and architecture, and
[CONTRIBUTING.md](CONTRIBUTING.md) for checks and constraints.

## Releases

[**v0.17.0**](https://github.com/numeric-lua/tensorflow-lua/releases/tag/v0.17.0)
ships the versioned rockspec and source rock. It is also published on
[LuaRocks](https://luarocks.org/modules/mateus-m-reis/tensorflow-lua). Release
notes are in [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Changelog and lint

See [CHANGELOG.md](CHANGELOG.md) for the milestone history. Run
`./scripts/lint.sh` for luacheck.

## Support

tensorflow-lua is free software under BSD-3-Clause. Donations are optional and
never unlock features or hold back builds.

<a href="https://github.com/sponsors/Mateus-M-Reis"><img src="https://img.shields.io/badge/Sponsor%20on%20GitHub-ea4aaa?logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"></a>

You can also star the repository, report issues, or read the organization
[support page](https://github.com/numeric-lua/.github/blob/main/SUPPORT.md).

## License

BSD-3-Clause. libtensorflow is downloaded separately under the Apache 2.0
license.
