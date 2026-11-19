local tf = require("tensorflow")

local function build_two_input()
  return function(inputs, graph)
    local weight = tf.variable(graph, 0.5, "weight")
    local sum = tf.add(graph, inputs.a, inputs.b, "input_sum")
    local output = tf.mul(graph, sum, weight.read, "sum", "float32")
    local difference = tf.sub(graph, inputs.a, inputs.b, "difference", "float32")
    return { sum = output, diff = difference }, { weight }
  end
end

test("functional model trains a two-input weighted sum and predicts", function()
  local model = tf.Functional.new({
    inputs = {
      { name = "a", shape = { 1 } },
      { name = "b", shape = { 1 } },
    },
    targets = { sum = { shape = { 1 } }, diff = { shape = { 1 } } },
    build = build_two_input(),
    optimizer = { type = "sgd", learning_rate = 0.001 },
  })
  local inputs = { a = { { 1 }, { 2 }, { 3 } }, b = { { 2 }, { 4 }, { 6 } } }
  local targets = { sum = { { 6 }, { 12 }, { 18 } },
    diff = { { -1 }, { -2 }, { -3 } } }
  local before = model:evaluate(inputs, targets)
  local history = model:fit(inputs, targets, { epochs = 200 })
  local after = model:evaluate(inputs, targets)
  assert_true(after < before)
  assert_true(#history.loss == 200)
  local predicted = model:predict({ a = { { 1 } }, b = { { 2 } } })
  assert_true(math.abs(predicted.sum:totable()[1][1] - 6) < 0.5)
  assert_true(math.abs(predicted.diff:totable()[1][1] - (-1)) < 1e-6)
  model:close()
end)

test("functional model checkpoints, summarizing, and callbacks", function()
  local function build(inputs, graph)
    local weight = tf.variable(graph, 0.5, "weight")
    local sum = tf.add(graph, inputs.a, inputs.b, "input_sum")
    return { sum = tf.mul(graph, sum, weight.read, "sum", "float32") },
      { weight }
  end
  local model = tf.Functional.new({
    inputs = { { name = "a", shape = { 1 } }, { name = "b", shape = { 1 } } },
    targets = { sum = { shape = { 1 } } },
    build = build,
    optimizer = { type = "sgd", learning_rate = 0.001 },
  })
  local x = { a = { { 1 }, { 2 } }, b = { { 2 }, { 4 } } }
  local y = { sum = { { 6 }, { 12 } } }
  local weights = model:get_weights()
  assert_eq(#weights, 1)
  model:set_weights(weights)
  local summary = model:summary()
  assert_true(summary:find("Total params") ~= nil)

  local dir = TMPDIR or "/tmp"
  local prefix = dir .. "/functional-checkpoint"
  model:save(prefix)
  model:load(prefix)
  local exported = dir .. "/functional.pb"
  model:export(exported)
  local file = assert(io.open(exported, "rb"))
  assert_true(#file:read("*a") > 0)
  file:close()

  local calls = 0
  local history = model:fit(x, y, {
    epochs = 3,
    callbacks = {
      { on_epoch_end = function(h) calls = calls + 1 end },
    },
  })
  assert_eq(#history.loss, 3)
  assert_eq(calls, 3)
  model:close()
end)
