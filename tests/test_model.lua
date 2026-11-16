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
