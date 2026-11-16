--
-- End-to-end graph-mode training: y = 2x + 1.
--
local root = (arg and arg[0] or "examples/linear_regression.lua"):
  match("^(.*)/examples/") or "."
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

local tf = require("tensorflow")

local graph = tf.Graph.new()
local x = tf.placeholder(graph, "float32", nil, "x")
local target = tf.placeholder(graph, "float32", nil, "target")
local w = tf.variable(graph, 0, "w")
local b = tf.variable(graph, 0, "b")

local prediction = tf.add(graph, tf.mul(graph, x, w.read, "wx"), b.read,
  "prediction")
local error = tf.sub(graph, prediction, target, "error")
local squared = tf.square(graph, error, "squared")
local axis = tf.constant(graph, { 0 }, "axis", "int32")
local mean = graph:new_operation("Mean", "loss")
mean:add_input(squared)
mean:add_input(axis)
mean:set_type("T", "float32")
mean:set_type("Tidx", "int32")
mean:set_bool("keep_dims", false)
local loss = mean:finish()
local updates = tf.train.gradient_descent(graph, loss, { w, b }, 0.05,
  "train")

local session = tf.Session.new(graph)
session:run({}, {}, { w.initializer, b.initializer })

local xs = { 0, 1, 2, 3 }
local ys = { 1, 3, 5, 7 }
for epoch = 1, 300 do
  session:run({ [x] = xs, [target] = ys }, {}, updates)
end
local result = session:run({}, { w.read, b.read })
print(string.format("w=%.4f b=%.4f", result[1]:totable(), result[2]:totable()))
assert(math.abs(result[1]:totable() - 2) < 0.15)
assert(math.abs(result[2]:totable() - 1) < 0.2)
session:close()
