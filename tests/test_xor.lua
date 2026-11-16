local tf = require("tensorflow")

test("trains an XOR MLP end to end", function()
  math.randomseed(7)
  local function random_matrix(rows, columns)
    local values = {}
    for r = 1, rows do
      values[r] = {}
      for c = 1, columns do values[r][c] = math.random() * 2 - 1 end
    end
    return values
  end
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 4, 2 }, "xx")
  local target = tf.placeholder(graph, "float32", { 4, 1 }, "xtarget")
  local w1 = tf.variable(graph, random_matrix(2, 4), "xw1")
  local b1 = tf.variable(graph, { 0, 0, 0, 0 }, "xb1")
  local w2 = tf.variable(graph, random_matrix(4, 1), "xw2")
  local b2 = tf.variable(graph, { 0 }, "xb2")
  local hidden = tf.nn.relu(graph,
    tf.add(graph, tf.matmul(graph, x, w1.read, "xh1"), b1.read, "xh1b"),
    "xhidden")
  local logits = tf.add(graph, tf.matmul(graph, hidden, w2.read, "xh2"),
    b2.read, "xlogits")
  local prediction = tf.nn.sigmoid(graph, logits, "xprediction")
  local loss = tf.reduce_mean(graph,
    tf.square(graph, tf.sub(graph, prediction, target, "xerror"), "xsquared"),
    { 0, 1 }, "xloss")
  local updates, _, state = tf.train.adam(graph, loss, { w1, b1, w2, b2 },
    { learning_rate = 0.1 })
  local session = tf.Session.new(graph)
  local init = { w1.initializer, b1.initializer, w2.initializer, b2.initializer }
  for _, value in ipairs(state.initializers) do init[#init + 1] = value end
  session:run({}, {}, init)
  local xs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
  local ys = { { 0 }, { 1 }, { 1 }, { 0 } }
  for _ = 1, 800 do session:run({ [x] = xs, [target] = ys }, {}, updates) end
  local output = session:run({ [x] = xs }, { prediction })[1]:totable()
  local correct = 0
  for i = 1, 4 do
    if (output[i][1] >= 0.5 and ys[i][1] == 1) or
       (output[i][1] < 0.5 and ys[i][1] == 0) then correct = correct + 1 end
  end
  assert_eq(correct, 4, "XOR accuracy")
  session:close()
end)
