local tf = require("tensorflow")

test("computes Huber, hinge, and categorical hinge losses", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 3 }, "pred")
  local targets = tf.placeholder(graph, "float32", { 3 }, "target")
  local huber = tf.losses.huber(graph, predictions, targets, { delta = 1 })
  local hinge = tf.losses.hinge(graph, predictions, targets, { margin = 1 })
  local session = tf.Session.new(graph)
  local value = session:run({
    [predictions] = { 0.5, 2, -1 }, [targets] = { -1, 1, 1 },
  }, { huber, hinge })
  assert_true(math.abs(value[1]:totable() - (1.0 + 0.5 + 1.5) / 3) < 1e-5)
  assert_true(math.abs(value[2]:totable() - (1.5 + 0 + 2) / 3) < 1e-5)
  session:close()

  local graph2 = tf.Graph.new()
  local logits = tf.placeholder(graph2, "float32", { 2, 3 }, "logits")
  local labels = tf.placeholder(graph2, "float32", { 2, 3 }, "labels")
  local categorical = tf.losses.categorical_hinge(graph2, logits, labels)
  local session2 = tf.Session.new(graph2)
  local loss = session2:run({
    [logits] = { { 0, 3, 1 }, { 2, 0, 1 } },
    [labels] = { { 1, 0, 0 }, { 1, 0, 0 } },
  }, { categorical })[1]:totable()
  assert_true(math.abs(loss - (4 + 0) / 2) < 1e-5)
  session2:close()
end)

test("weights dense softmax cross entropy by class", function()
  local graph = tf.Graph.new()
  local logits = tf.placeholder(graph, "float32", { 2, 2 }, "logits")
  local labels = tf.placeholder(graph, "float32", { 2, 2 }, "labels")
  local loss = tf.losses.softmax_cross_entropy(graph, logits, labels, {
    per_example = false, class_weights = { [0] = 1, [1] = 3 },
  })
  local session = tf.Session.new(graph)
  local value = session:run({
    [logits] = { { 1, 0 }, { 0, 1 } },
    [labels] = { { 1, 0 }, { 0, 1 } },
  }, { loss })[1]:totable()
  assert_true(value > 0)
  session:close()
end)

test("computes precision, recall, and F1", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 4 }, "prob")
  local labels = tf.placeholder(graph, "float32", { 4 }, "labels")
  local precision = tf.metrics.precision(graph, predictions, labels,
    { name = "precision" })
  local recall = tf.metrics.recall(graph, predictions, labels,
    { name = "recall" })
  local f1 = tf.metrics.f1(graph, predictions, labels, { name = "f1" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [predictions] = { 0.9, 0.8, 0.2, 0.1 },
    [labels] = { 1, 0, 1, 0 },
  }, { precision, recall, f1 })
  assert_true(math.abs(values[1]:totable() - 0.5) < 1e-5)
  assert_true(math.abs(values[2]:totable() - 0.5) < 1e-5)
  assert_true(math.abs(values[3]:totable() - 0.5) < 1e-5)
  session:close()
end)

test("computes macro multiclass precision, recall, and F1", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "int64", { 4 }, "class_pred")
  local labels = tf.placeholder(graph, "int32", { 4 }, "class_label")
  local precision, recall, f1, per_class = tf.metrics.multiclass_prf(
    graph, predictions, labels, 3, { name = "macro" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [predictions] = { 0, 1, 2, 0 },
    [labels] = { 0, 1, 2, 1 },
  }, { precision, recall, f1, per_class.precision[1] })
  assert_true(math.abs(values[1]:totable() - (5 / 6)) < 1e-5)
  assert_true(math.abs(values[2]:totable() - (5 / 6)) < 1e-5)
  assert_true(math.abs(values[3]:totable() - (7 / 9)) < 1e-5)
  assert_true(math.abs(values[4]:totable() - 1.0) < 1e-5)
  session:close()
end)

test("computes binary ROC AUC including ties", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 4 }, "auc_pred")
  local labels = tf.placeholder(graph, "float32", { 4 }, "auc_label")
  local auc = tf.metrics.auc(graph, predictions, labels, { name = "auc" })
  local session = tf.Session.new(graph)
  local perfect = session:run({
    [predictions] = { 0.9, 0.8, 0.2, 0.1 },
    [labels] = { 1, 1, 0, 0 },
  }, { auc })[1]:totable()
  assert_true(math.abs(perfect - 1) < 1e-6)
  local reversed = session:run({
    [predictions] = { 0.1, 0.2, 0.8, 0.9 },
    [labels] = { 1, 1, 0, 0 },
  }, { auc })[1]:totable()
  assert_true(math.abs(reversed) < 1e-6)
  session:close()
end)
