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
