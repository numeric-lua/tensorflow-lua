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

test("computes classification accuracy", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "int32", { 3 }, "acc_pred")
  local labels = tf.placeholder(graph, "int32", { 3 }, "acc_true")
  local accuracy = tf.metrics.accuracy(graph, predictions, labels)
  local session = tf.Session.new(graph)
  local value = session:run({
    [predictions] = tf.tensor({ 1, 0, 1 }, { dtype = "int32" }),
    [labels] = tf.tensor({ 1, 1, 1 }, { dtype = "int32" }),
  }, { accuracy })[1]:totable()
  assert_true(math.abs(value - 2 / 3) < 1e-6)
  session:close()
end)

test("computes binary cross-entropy", function()
  local graph = tf.Graph.new()
  local logits = tf.placeholder(graph, "float32", { 2 }, "bce_logits")
  local labels = tf.placeholder(graph, "float32", { 2 }, "bce_labels")
  local loss = tf.losses.binary_cross_entropy(graph, logits, labels)
  local session = tf.Session.new(graph)
  local value = session:run({ [logits] = { 0, 0 }, [labels] = { 1, 1 } },
    { loss })[1]:totable()
  assert_true(math.abs(value - math.log(2)) < 1e-6)
  session:close()
end)

test("computes mean absolute error", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 2 }, "mae_pred")
  local targets = tf.placeholder(graph, "float32", { 2 }, "mae_true")
  local loss = tf.losses.mean_absolute_error(graph, predictions, targets,
    { axes = { 0 } })
  local session = tf.Session.new(graph)
  local value = session:run({ [predictions] = { 1, -2 }, [targets] = { 0, 0 } },
    { loss })[1]:totable()
  assert_true(math.abs(value - 1.5) < 1e-6)
  session:close()
end)
