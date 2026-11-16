local tf = require("tensorflow")

test("builds and runs a dense layer", function()
  math.randomseed(11)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 3 }, "dense_input")
  local layer = tf.layers.dense(graph, input, 4, { input_size = 3,
    activation = "relu", name = "fc" })
  assert_eq(#layer.variables, 2)
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local output = session:run({ [input] = { { 1, 2, 3 }, { 4, 5, 6 } } },
    { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "2,4")
  session:close()
end)

test("builds a sequential stack", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2 }, "seq_input")
  local layers = tf.layers.sequential(graph, input, {
    { units = 3, options = { activation = "relu" } },
    { units = 1, options = {} },
  }, 2)
  assert_eq(#layers, 2)
  assert_eq(layers[1].input_size, 2)
  assert_eq(layers[2].input_size, 3)
  assert_eq(layers[2].output.index, 0)
end)

test("dense validates input size", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", nil, "bad_dense")
  assert_error(function() tf.layers.dense(graph, input, 4, {}) end,
    "input_size")
end)
