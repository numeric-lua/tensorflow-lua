# TensorFlow in a remote notebook kernel

`tensorflow-lua` runs inside the LuaJIT gateway from
[lua-notebooks](https://github.com/numeric-lua/lua-notebooks). The browser is
only a front-end; TensorFlow stays on the kernel machine.

## Start the gateway

```sh
export TF_LUA=/absolute/path/to/tensorflow-lua
export LUA_PATH="$TF_LUA/?.lua;$TF_LUA/?/init.lua;;"
export TENSORFLOW_ROOT="$TF_LUA/third_party/libtensorflow"
luajit /absolute/path/to/lua-notebooks/scripts/kernel-gateway.lua --port 8765
```

Point `remote.txt` in the LÖVR web project at:

```text
ws://127.0.0.1:8765
```

## Cell

```lua
local tf = require("tensorflow")
local model = tf.Sequential.new({ input_shape = { 1 } })
model:add({ units = 1 })
model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
model:fit({ { 1 }, { 2 } }, { { 3 }, { 5 } }, { epochs = 1 })
print("tensorflow", tf.VERSION)
return tf.VERSION
```

Verified end to end against a local gateway: the cell trained, printed
`tensorflow 0.10.0`, and returned `0.10.0` to the client.

Full LÖVR notebook rendering is still blocked on the upstream WebGPU issue
documented in the lua-notebooks web bridge; the remote transport and kernel are
independent of that rendering problem.
