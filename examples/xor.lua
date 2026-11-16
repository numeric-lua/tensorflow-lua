--
-- Graph-mode XOR MLP: 2 -> 4 (ReLU) -> 1 (sigmoid), trained with Adam.
--
local root = (arg and arg[0] or "examples/xor.lua"):match("^(.*)/examples/") or "."
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

local tf = require("tensorflow")
math.randomseed(7)

local function random_matrix(rows, columns, scale)
  local values = {}
  for r = 1, rows do
    values[r] = {}
    for c = 1, columns do
      values[r][c] = (math.random() * 2 - 1) * scale
    end
  end
  return values
end

local graph = tf.Graph.new()
local x = tf.placeholder(graph, "float32", { 4, 2 }, "x")
local target = tf.placeholder(graph, "float32", { 4, 1 }, "target")
local w1 = tf.variable(graph, random_matrix(2, 4, 1), "w1")
local b1 = tf.variable(graph, { 0, 0, 0, 0 }, "b1")
local w2 = tf.variable(graph, random_matrix(4, 1, 1), "w2")
local b2 = tf.variable(graph, { 0 }, "b2")

local hidden = tf.nn.relu(graph,
  tf.add(graph, tf.matmul(graph, x, w1.read, "h1"), b1.read, "h1_bias"),
  "hidden")
local logits = tf.add(graph, tf.matmul(graph, hidden, w2.read, "h2"),
  b2.read, "logits")
local prediction = tf.nn.sigmoid(graph, logits, "prediction")
local error = tf.sub(graph, prediction, target, "error")
local squared = tf.square(graph, error, "squared")
local loss = tf.reduce_mean(graph, squared, { 0, 1 }, "loss")
local updates, _, state = tf.train.adam(graph, loss, { w1, b1, w2, b2 },
  { learning_rate = 0.1 })

local session = tf.Session.new(graph)
local init = { w1.initializer, b1.initializer, w2.initializer, b2.initializer }
for _, value in ipairs(state.initializers) do init[#init + 1] = value end
session:run({}, {}, init)

local xs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
local ys = { { 0 }, { 1 }, { 1 }, { 0 } }
for _ = 1, 800 do
  session:run({ [x] = xs, [target] = ys }, {}, updates)
end
local output = session:run({ [x] = xs }, { prediction })[1]:totable()
local correct = 0
for i = 1, 4 do
  if (output[i][1] >= 0.5 and ys[i][1] == 1) or
     (output[i][1] < 0.5 and ys[i][1] == 0) then correct = correct + 1 end
end
print(string.format("xor accuracy: %d/4", correct))
assert(correct == 4)
session:close()
