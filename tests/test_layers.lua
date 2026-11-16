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

test("builds and runs an embedding layer", function()
  math.randomseed(15)
  local graph = tf.Graph.new()
  local indices = tf.placeholder(graph, "int32", { 2 }, "emb_indices")
  local layer = tf.layers.embedding(graph, indices, 5, 3, { name = "emb" })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local output = session:run({ [indices] = { 0, 4 } }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "2,3")
  session:close()
end)

test("flattens per-sample features", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 2, 2 }, "flat_input")
  local layer = tf.layers.flatten(graph, input, 4, { name = "flat" })
  local session = tf.Session.new(graph)
  local values = { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }
  local output = session:run({ [input] = values }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "2,4")
  assert_eq(output:totable()[2][4], 8)
  session:close()
end)

test("flatten validates input size", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1 }, "flat_bad")
  assert_error(function() tf.layers.flatten(graph, input, 0) end, "input_size")
end)

test("dropout with zero rate is identity", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 2 }, "drop_input")
  local layer = tf.layers.dropout(graph, input, 0, { training = true,
    seed = 1 })
  local session = tf.Session.new(graph)
  local values = { { 1, 2 }, { 3, 4 } }
  local output = session:run({ [input] = values }, { layer.output })[1]
  assert_eq(output:totable()[2][1], 3)
  session:close()
end)

test("dropout in inference mode returns input", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1 }, "drop_eval")
  local layer = tf.layers.dropout(graph, input, 0.5, { training = false })
  assert_true(layer.output == input)
end)

test("dropout masks or scales values", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 4 }, "drop_mask_input")
  local layer = tf.layers.dropout(graph, input, 0.5, { training = true,
    seed = 3 })
  local session = tf.Session.new(graph)
  local values = { 1, 2, 3, 4 }
  local output = session:run({ [input] = values }, { layer.output })[1]:totable()
  assert_eq(#output, 4)
  for i = 1, 4 do
    assert_true(output[i] == 0 or math.abs(output[i] - values[i] * 2) < 1e-5,
      tostring(output[i]))
  end
  session:close()
end)

test("runs a Conv2D layer", function()
  math.randomseed(28)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 1 }, "conv_input")
  local layer = tf.layers.conv2d(graph, input, 2, 3,
    { in_channels = 1, padding = "SAME", name = "conv" })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local values = {}
  for y = 1, 4 do
    values[y] = {}
    for x = 1, 4 do values[y][x] = { y + x } end
  end
  local output = session:run({ [input] = { values } }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,4,4,2")
  session:close()
end)

test("runs a MaxPool2D layer", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 2 }, "pool_input")
  local layer = tf.layers.max_pool2d(graph, input, 2, { stride = 2 })
  local session = tf.Session.new(graph)
  local values = {}
  for y = 1, 4 do
    values[y] = {}
    for x = 1, 4 do values[y][x] = { y, x } end
  end
  local output = session:run({ [input] = { values } }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,2,2,2")
  session:close()
end)
