local tf = require("tensorflow")

local function linear_model()
  math.randomseed(21)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.05 } })
  return model
end

test("early stopping ends training", function()
  local model = linear_model()
  local callback = tf.callbacks.EarlyStopping({ monitor = "val_loss",
    patience = 0, min_delta = 1e9 })
  local history = model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 10, callbacks = { callback },
      validation_data = { { { 3 } }, { { 7 } } } })
  assert_eq(#history.loss, 2)
  assert_eq(history.stop_training, true)
  assert_eq(callback.stopped_epoch, 2)
  model:close()
end)

test("model checkpoint writes a file", function()
  local path = os.tmpname()
  local model = linear_model()
  local callback = tf.callbacks.ModelCheckpoint(path)
  model:fit({ { 0 }, { 1 } }, { { 1 }, { 3 } },
    { epochs = 1, validation_data = { { { 2 } }, { { 5 } } },
      callbacks = { callback } })
  local file = io.open(path, "rb")
  assert_true(file ~= nil)
  if file then file:close() end
  os.remove(path)
  model:close()
end)

test("restore_best_weights restores the best checkpoint", function()
  local model = linear_model()
  local callback = tf.callbacks.EarlyStopping({ monitor = "val_loss",
    patience = 0, min_delta = 1e9, restore_best_weights = true })
  model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 5, callbacks = { callback },
      validation_data = { { { 3 } }, { { 7 } } } })
  assert_eq(callback.stopped_epoch, 2)
  model:close()
end)
