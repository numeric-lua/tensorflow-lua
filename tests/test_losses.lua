local tf = require("tensorflow")

test("computes mean squared error", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 2 }, "mse_pred")
  local targets = tf.placeholder(graph, "float32", { 2 }, "mse_true")
  local loss = tf.losses.mean_squared_error(graph, predictions, targets,
    { axes = { 0 } })
  local session = tf.Session.new(graph)
  local value = session:run({ [predictions] = { 1, 2 }, [targets] = { 0, 0 } },
    { loss })[1]:totable()
  assert_true(math.abs(value - 2.5) < 1e-6)
  session:close()
end)

test("computes softmax cross-entropy", function()
  local graph = tf.Graph.new()
  local logits = tf.placeholder(graph, "float32", { 1, 2 }, "ce_logits")
  local labels = tf.placeholder(graph, "float32", { 1, 2 }, "ce_labels")
  local loss = tf.losses.softmax_cross_entropy(graph, logits, labels)
  local session = tf.Session.new(graph)
  local value = session:run({ [logits] = { { 0, 0 } },
    [labels] = { { 1, 0 } } }, { loss })[1]:totable()
  assert_true(math.abs(value - math.log(2)) < 1e-6)
  session:close()
end)
