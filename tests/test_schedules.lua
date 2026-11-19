local tf = require("tensorflow")

test("exposes exponential, cosine, polynomial, and piecewise schedules", function()
  local exponential = tf.schedules.exponential_decay(1, 10, 0.5)
  assert_true(math.abs((exponential(0)) - (1)) <= (0))
  assert_true(math.abs((exponential(10)) - (0.5)) <= (1e-9))
  local staircase = tf.schedules.exponential_decay(1, 10, 0.5, true)
  assert_true(math.abs((staircase(9)) - (1)) <= (0))
  assert_true(math.abs((staircase(10)) - (0.5)) <= (0))

  local cosine = tf.schedules.cosine_decay(1, 10)
  assert_true(math.abs((cosine(0)) - (1)) <= (1e-9))
  assert_true(math.abs((cosine(10)) - (0)) <= (1e-9))

  local polynomial = tf.schedules.polynomial_decay(1, 10, 0, 2)
  assert_true(math.abs((polynomial(0)) - (1)) <= (0))
  assert_true(math.abs((polynomial(10)) - (0)) <= (0))

  local piecewise = tf.schedules.piecewise_constant(1, { 5, 10 }, { 1, 0.5, 0.1 })
  assert_true(math.abs((piecewise(4)) - (1)) <= (0))
  assert_true(math.abs((piecewise(5)) - (0.5)) <= (0))
  assert_true(math.abs((piecewise(20)) - (0.1)) <= (0))
end)

test("Sequential accepts a schedule as its learning rate", function()
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({
    optimizer = { type = "sgd",
      learning_rate = tf.schedules.exponential_decay(0.1, 2, 0.5) },
    loss = "mse",
  })
  local x = tf.tensor({ { 1 }, { 2 }, { 3 } })
  local y = tf.tensor({ { 2 }, { 4 }, { 6 } })
  local history = model:fit_tensor(x, y, { epochs = 3, batch_size = 2 })
  assert_eq(#history.loss, 3)
  x:destroy(); y:destroy(); model:close()
end)
