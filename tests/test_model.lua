local tf = require("tensorflow")

test("trains a Sequential regression model", function()
  math.randomseed(3)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.1 },
    loss = "mse" })
  local x = { { 0 }, { 1 }, { 2 }, { 3 } }
  local y = { { 1 }, { 3 }, { 5 }, { 7 } }
  local history = model:fit(x, y, { epochs = 200, batch_size = 4 })
  assert_eq(#history.loss, 200)
  assert_true(history.loss[200] < history.loss[1])
  local prediction = model:predict({ { 4 } }):totable()[1][1]
  assert_true(math.abs(prediction - 9) < 0.5, tostring(prediction))
  model:close()
end)

test("validates model configuration", function()
  assert_error(function() tf.Sequential.new({}) end, "input_shape")
  local model = tf.Sequential.new({ input_shape = { 1 } })
  assert_error(function() model:add({}) end, "units")
  model:add({ units = 1 })
  model:compile({ loss = "unknown" })
  assert_error(function() model:fit({ { 1 } }, { { 1 } }) end, "loss")
end)

test("saves and restores Sequential weights", function()
  math.randomseed(4)
  local path = os.tmpname()
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.1 } })
  model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 100 })
  local before = model:predict({ { 3 } }):totable()[1][1]
  model:save(path)
  model:close()

  local restored = tf.Sequential.new({ input_shape = { 1 } })
  restored:add({ units = 1 })
  restored:compile({ optimizer = { type = "adam", learning_rate = 0.1 } })
  restored:load(path)
  local after = restored:predict({ { 3 } }):totable()[1][1]
  assert_true(math.abs(before - after) < 1e-4,
    tostring(before) .. " vs " .. tostring(after))
  restored:close()
  os.remove(path)
end)

test("reports validation loss and runs epoch callbacks", function()
  math.randomseed(5)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.05 } })
  local epochs = 0
  local history = model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 5, callbacks = { { on_epoch_end = function()
      epochs = epochs + 1
    end } }, validation_data = { { { 3 } }, { { 7 } } } })
  assert_eq(epochs, 5)
  assert_eq(#history.val_loss, 5)
  model:close()
end)

test("exports a Sequential GraphDef", function()
  math.randomseed(6)
  local path = os.tmpname()
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 3 })
  model:compile()
  model:export(path)
  local graph = tf.Graph.new():import_file(path)
  assert_true(graph:operation("input") ~= nil)
  model:close()
  os.remove(path)
end)

test("saves and restores native TensorFlow model checkpoints", function()
  math.randomseed(7)
  local prefix = "/tmp/tf-lua-model-v2/ckpt"
  os.execute("rm -rf /tmp/tf-lua-model-v2")
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.1 } })
  model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } }, { epochs = 50 })
  local before = model:predict({ { 3 } }):totable()[1][1]
  model:save_native(prefix)
  model:close()

  local restored = tf.Sequential.new({ input_shape = { 1 } })
  restored:add({ units = 1 })
  restored:compile({ optimizer = { type = "adam", learning_rate = 0.1 } })
  restored:load_native(prefix)
  local after = restored:predict({ { 3 } }):totable()[1][1]
  assert_true(math.abs(before - after) < 1e-4)
  restored:close()
  os.execute("rm -rf /tmp/tf-lua-model-v2")
end)

test("reports accuracy during classification training", function()
  math.randomseed(8)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 8, options = { activation = "relu" } })
  model:add({ units = 2 })
  model:compile({ loss = "softmax_cross_entropy",
    optimizer = { type = "adam", learning_rate = 0.05 },
    metrics = { "accuracy" } })
  local x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
  local y = { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } }
  local history = model:fit(x, y, { epochs = 150, batch_size = 4 })
  assert_eq(#history.accuracy, 150)
  assert_true(history.accuracy[150] >= 0.5)
  local loss, metrics = model:evaluate(x, y)
  assert_true(loss >= 0)
  assert_true(metrics.accuracy >= 0 and metrics.accuracy <= 1)
  model:close()
end)

test("rejects empty datasets and invalid layers", function()
  local model = tf.Sequential.new({ input_shape = { 1 } })
  assert_error(function() model:add({ units = 0 }) end, "positive")
  model:add({ units = 1 })
  model:compile()
  assert_error(function() model:fit({}, {}) end, "non-empty")
end)

test("supports shuffled training", function()
  math.randomseed(9)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local history = model:fit({ { 0 }, { 1 }, { 2 }, { 3 } },
    { { 1 }, { 3 }, { 5 }, { 7 } }, { epochs = 3, batch_size = 2,
      shuffle = true })
  assert_eq(#history.loss, 3)
  model:close()
end)

test("trains with binary cross-entropy", function()
  math.randomseed(10)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ loss = "bce",
    optimizer = { type = "adam", learning_rate = 0.1 },
    metrics = { "binary_accuracy" } })
  local history = model:fit({ { 0 }, { 1 }, { 2 }, { 3 } },
    { { 0 }, { 0 }, { 1 }, { 1 } }, { epochs = 100 })
  assert_eq(#history.loss, 100)
  assert_eq(#history.binary_accuracy, 100)
  local prediction = model:predict({ { 3 } }):totable()[1][1]
  assert_true(prediction > 0.5, tostring(prediction))
  model:close()
end)

test("trains with sparse integer labels", function()
  math.randomseed(11)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce",
    optimizer = { type = "adam", learning_rate = 0.05 },
    metrics = { "accuracy" } })
  local x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
  local y = { 0, 1, 1, 0 }
  local history = model:fit(x, y, { epochs = 100, batch_size = 4 })
  assert_eq(#history.loss, 100)
  assert_eq(#history.accuracy, 100)
  local _, metrics = model:evaluate(x, y)
  assert_true(metrics.accuracy >= 0.5)
  model:close()
end)

test("trains directly from a Dataset", function()
  math.randomseed(12)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local dataset = tf.data.Dataset.new({ { 0 }, { 1 }, { 2 }, { 3 } },
    { { 1 }, { 3 }, { 5 }, { 7 } }):shuffle(3):batch(2)
  local history = model:fit_dataset(dataset, { epochs = 3 })
  assert_eq(#history.loss, 3)
  model:close()
end)

test("trains a Sequential model with AdamW", function()
  math.randomseed(13)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adamw", learning_rate = 0.1,
    weight_decay = 0.01 } })
  local history = model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 50 })
  assert_eq(#history.loss, 50)
  model:close()
end)

test("fit accepts a Dataset object directly", function()
  math.randomseed(14)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local dataset = tf.data.Dataset.new({ { 0 }, { 1 }, { 2 }, { 3 } },
    { { 1 }, { 3 }, { 5 }, { 7 } }):batch(2)
  local history = model:fit(dataset, { epochs = 2 })
  assert_eq(#history.loss, 2)
  model:close()
end)

test("uses Dataset validation and prediction paths", function()
  math.randomseed(16)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local dataset = tf.data.Dataset.new({ { 0 }, { 1 }, { 2 }, { 3 } },
    { { 1 }, { 3 }, { 5 }, { 7 } })
  local train, validation = dataset:split(0.75)
  local history = model:fit(train, { epochs = 2, validation_data = validation })
  assert_eq(#history.val_loss, 2)
  local loss, metrics = model:evaluate_dataset(validation)
  assert_true(loss >= 0)
  assert_true(metrics ~= nil)
  local prediction = model:predict_dataset(validation)
  assert_eq(prediction:shape()[1], 1)
  model:close()
end)
