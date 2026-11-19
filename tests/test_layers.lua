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

test("runs a Conv2DTranspose layer with deterministic weights", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 1 },
    "deconv_input")
  local layer = tf.layers.conv2d_transpose(graph, input, 1, 2,
    { in_channels = 1, stride = 1, padding = "VALID", use_bias = false,
      output_shape = { 1, 3, 3, 1 }, name = "deconv" })
  assert_eq(table.concat(layer.weight.shape, ","), "2,2,1,1")
  local weight_values = {
    { { { 1 } }, { { 2 } } },
    { { { 3 } }, { { 4 } } },
  }
  local assign = graph:new_operation("Assign", "set_deconv_weight")
  assign:add_input(layer.weight.ref)
  assign:add_input(tf.constant(graph, weight_values, "deconv_weight",
    "float32"))
  assign:set_type("T", "float32")
  assign:set_bool("use_locking", true)
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  session:run({}, {}, { assign:finish() })
  local values = { { { { 1 }, { 1 } }, { { 1 }, { 1 } } } }
  local output = session:run({ [input] = values }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,3,3,1")
  local expected = { { 1, 3, 2 }, { 4, 10, 6 }, { 3, 7, 4 } }
  local actual = output:totable()[1]
  for y = 1, 3 do
    for x = 1, 3 do
      assert_true(math.abs(actual[y][x][1] - expected[y][x]) < 1e-6,
        string.format("output[%d][%d]=%f", y, x, actual[y][x][1]))
    end
  end
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

test("runs BatchNormalization in training and inference modes", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 4, 3 }, "bn_input")
  local training = tf.placeholder(graph, "bool", { }, "bn_training")
  local layer = tf.layers.batch_normalization(graph, input, 3,
    { training = training, axes = { 0 }, name = "bn" })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local values = { { 1, 2, 3 }, { 2, 3, 4 }, { 3, 4, 5 }, { 4, 5, 6 } }
  local trained = session:run({ [input] = values, [training] = true },
    { layer.output })[1]
  assert_eq(table.concat(trained:shape(), ","), "4,3")
  local inferred = session:run({ [input] = values, [training] = false },
    { layer.output })[1]
  assert_eq(table.concat(inferred:shape(), ","), "4,3")
  local mean = 0
  for _, row in ipairs(trained:totable()) do
    for _, value in ipairs(row) do mean = mean + value end
  end
  assert_true(math.abs(mean) < 1e-3, "mean=" .. tostring(mean))
  session:close()
end)

test("runs average and global average pooling", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 2 }, "avg_input")
  local avg = tf.layers.avg_pool2d(graph, input, 2, { stride = 2 })
  local global_avg = tf.layers.global_avg_pool2d(graph, input,
    { name = "global" })
  local session = tf.Session.new(graph)
  local values = {}
  for y = 1, 4 do
    values[y] = {}
    for x = 1, 4 do values[y][x] = { y, x } end
  end
  local pooled = session:run({ [input] = { values } }, { avg.output })[1]
  assert_eq(table.concat(pooled:shape(), ","), "1,2,2,2")
  local global_values = session:run({ [input] = { values } },
    { global_avg.output })[1]
  assert_eq(table.concat(global_values:shape(), ","), "1,2")
  session:close()
end)

test("runs a Conv1D layer", function()
  math.randomseed(36)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 5, 2 }, "conv1_input")
  local layer = tf.layers.conv1d(graph, input, 3, 3,
    { in_channels = 2, padding = "SAME", stride = 2 })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local values = {}
  for s = 1, 5 do values[s] = { s, s + 1 } end
  local output = session:run({ [input] = { values } }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,3,3")
  session:close()
end)

test("runs global max pooling and activation layers", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 2 }, "max_input")
  local max = tf.layers.global_max_pool2d(graph, input, { name = "gmax" })
  local relu = tf.layers.activation(graph, max.output, "relu",
    { name = "relu" })
  local session = tf.Session.new(graph)
  local values = {}
  for y = 1, 4 do
    values[y] = {}
    for x = 1, 4 do values[y][x] = { y, -x } end
  end
  local pooled = session:run({ [input] = { values } }, { relu.output })[1]
  assert_eq(table.concat(pooled:shape(), ","), "1,2")
  local flat = pooled:totable()[1]
  assert_eq(flat[1], 4)
  assert_eq(flat[2], 0)
  session:close()
end)

test("runs reshape and permute layers", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 6 }, "shape_input")
  local reshape = tf.layers.reshape(graph, input, { 2, 2, 3 },
    { name = "reshape" })
  local permute = tf.layers.permute(graph, reshape.output, { 0, 2, 1 },
    { name = "permute" })
  local session = tf.Session.new(graph)
  local values = { { 1, 2, 3, 4, 5, 6 }, { 7, 8, 9, 10, 11, 12 } }
  local output = session:run({ [input] = values }, { permute.output })[1]
  assert_eq(table.concat(output:shape(), ","), "2,3,2")
  session:close()
end)

test("runs padding, upsampling, and depthwise convolution", function()
  math.randomseed(43)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 2 }, "diff_input")
  local padded = tf.layers.zero_padding2d(graph, input, 1,
    { name = "pad" })
  local depthwise = tf.layers.depthwise_conv2d(graph, padded.output, 2, 3,
    { in_channels = 2, padding = "SAME", name = "depthwise" })
  local upsampled = tf.layers.upsampling2d(graph, depthwise.output, 2,
    { input_shape = { 6, 6, 4 }, name = "upsample" })
  local session = tf.Session.new(graph)
  session:run({}, {}, depthwise.initializers)
  local values = {}
  for y = 1, 4 do
    values[y] = {}
    for x = 1, 4 do values[y][x] = { y, x } end
  end
  local output = session:run({ [input] = { values } },
    { upsampled.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,12,12,4")
  session:close()
end)

test("runs global max-pooling over time", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 5, 3 }, "gm1_input")
  local layer = tf.layers.global_max_pool1d(graph, input, { name = "gmp1" })
  local session = tf.Session.new(graph)
  local values = {}
  for n = 1, 2 do
    values[n] = {}
    for t = 1, 5 do values[n][t] = { t, -t, n } end
  end
  local output = session:run({ [input] = values }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "2,3")
  local flat = output:totable()
  assert_eq(flat[1][1], 5)
  assert_eq(flat[1][2], -1)
  session:close()
end)

test("runs 1-D pooling and global average pooling", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 2 }, "pool1d")
  local pooled = tf.layers.max_pool1d(graph, input, 2, { name = "max1" })
  local average = tf.layers.avg_pool1d(graph, input, 2,
    { name = "avg1" })
  local global = tf.layers.global_avg_pool1d(graph, input,
    { name = "global1" })
  local session = tf.Session.new(graph)
  local values = { { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } } }
  local results = session:run({ [input] = values },
    { pooled.output, average.output, global.output })
  local first, second, third = results[1], results[2], results[3]
  assert_eq(table.concat(first:shape(), ","), "1,2,2")
  assert_eq(table.concat(second:shape(), ","), "1,2,2")
  assert_eq(table.concat(third:shape(), ","), "1,2")
  assert_eq(first:totable()[1][1][1], 3)
  session:close()
end)

test("runs separable convolution and layer normalization", function()
  local graph = tf.Graph.new()
  local image = tf.placeholder(graph, "float32", { 1, 4, 4, 2 }, "sep_in")
  local layer = tf.layers.separable_conv2d(graph, image, 3, 3, {
    in_channels = 2, padding = "SAME", name = "sep",
  })
  local flat = tf.placeholder(graph, "float32", { 2, 4 }, "ln_in")
  local norm = tf.layers.layer_normalization(graph, flat, {
    normalized_shape = { 4 }, name = "ln",
  })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  session:run({}, {}, norm.initializers)
  local image_values = {}
  for y = 1, 4 do
    image_values[y] = {}
    for x = 1, 4 do image_values[y][x] = { y, x } end
  end
  local output = session:run({ [image] = { image_values } },
    { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,4,4,3")
  local normalized = session:run({ [flat] = { { 1, 2, 3, 4 }, { 2, 4, 6, 8 } } },
    { norm.output })[1]:totable()
  local mean = (normalized[1][1] + normalized[1][2] + normalized[1][3] +
    normalized[1][4]) / 4
  assert_true(math.abs(mean) < 1e-5)
  session:close()
end)

test("embedding padding_idx returns a zero row", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "int32", { 3 }, "embed_idx")
  local layer = tf.layers.embedding(graph, input, 5, 4, {
    padding_idx = 2, name = "embed",
  })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local output = session:run({ [input] = { 0, 2, 3 } },
    { layer.output })[1]:totable()
  for column = 1, 4 do assert_eq(output[2][column], 0) end
  session:close()
end)

test("runs 3-D convolution, pooling, and global pooling", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 4, 2 }, "vol")
  local conv = tf.layers.conv3d(graph, input, 3, 3, {
    in_channels = 2, input_shape = { 4, 4, 4, 2 }, padding = "SAME",
    name = "conv3",
  })
  local pool = tf.layers.max_pool3d(graph, conv.output, 2,
    { name = "pool3" })
  local global_avg = tf.layers.global_avg_pool3d(graph, conv.output,
    { name = "global3" })
  local session = tf.Session.new(graph)
  session:run({}, {}, conv.initializers)
  local values = {}
  for d = 1, 4 do
    values[d] = {}
    for y = 1, 4 do
      values[d][y] = {}
      for x = 1, 4 do values[d][y][x] = { d / 4, y / 4 } end
    end
  end
  local outputs = session:run({ [input] = { values } },
    { conv.output, pool.output, global_avg.output })
  assert_eq(table.concat(outputs[1]:shape(), ","), "1,4,4,4,3")
  assert_eq(table.concat(outputs[2]:shape(), ","), "1,2,2,2,3")
  assert_eq(table.concat(outputs[3]:shape(), ","), "1,3")
  session:close()
end)

test("composed Conv3D honors VALID padding and strides", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 5, 5, 5, 2 }, "valid_vol")
  local layer = tf.layers.conv3d(graph, input, 2, 3, {
    in_channels = 2, input_shape = { 5, 5, 5, 2 }, padding = "VALID",
    stride = 2, name = "valid_conv3",
  })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local volume = {}
  for d = 1, 5 do
    volume[d] = {}
    for y = 1, 5 do
      volume[d][y] = {}
      for x = 1, 5 do volume[d][y][x] = { 1, 0 } end
    end
  end
  local output = session:run({ [input] = { volume } },
    { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,2,2,2,2")
  session:close()
end)

test("scaled dot-product attention returns context and weights", function()
  local graph = tf.Graph.new()
  local query = tf.placeholder(graph, "float32", { 1, 4, 8 }, "q")
  local key = tf.placeholder(graph, "float32", { 1, 4, 8 }, "k")
  local value = tf.placeholder(graph, "float32", { 1, 4, 8 }, "v")
  local output, weights = tf.layers.attention(graph, query, key, value, {
    query_shape = { 4, 8 }, key_shape = { 4, 8 }, causal = true,
    name = "attn",
  })
  local session = tf.Session.new(graph)
  local values = {}
  for i = 1, 4 do values[i] = {}; for j = 1, 8 do values[i][j] = i + j end end
  local result = session:run({
    [query] = { values }, [key] = { values }, [value] = { values },
  }, { output, weights })
  assert_eq(table.concat(result[1]:shape(), ","), "1,4,8")
  assert_eq(table.concat(result[2]:shape(), ","), "1,4,4")
  local first = result[2]:totable()[1][1]
  assert_true(math.abs(first[2]) < 1e-6)
  assert_true(math.abs(first[3]) < 1e-6)
  session:close()
end)

test("multi-head attention is trainable", function()
  local graph = tf.Graph.new()
  local query = tf.placeholder(graph, "float32", { 1, 4, 8 }, "mha_q")
  local key = tf.placeholder(graph, "float32", { 1, 4, 8 }, "mha_k")
  local value = tf.placeholder(graph, "float32", { 1, 4, 8 }, "mha_v")
  local layer = tf.layers.multi_head_attention(graph, query, key, value, {
    query_shape = { 4, 8 }, key_shape = { 4, 8 },
    num_heads = 2, key_dim = 4, output_dim = 8, name = "mha",
  })
  local loss = tf.reduce_sum(graph, layer.output, { 0, 1, 2 }, "mha_loss",
    "float32")
  local refs = {}
  for index, variable in ipairs(layer.variables) do refs[index] = variable.ref end
  local gradients = tf.gradients(graph, { loss }, refs)
  assert_eq(#gradients, #refs)
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local values = {}
  for i = 1, 4 do values[i] = {}; for j = 1, 8 do values[i][j] = i * j / 8 end end
  local result = session:run({
    [query] = { values }, [key] = { values }, [value] = { values },
  }, { layer.output, gradients[1] })
  assert_eq(table.concat(result[1]:shape(), ","), "1,4,8")
  assert_true(result[2] ~= nil)
  session:close()
end)

test("positional encoding and transformer block are trainable", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 8 }, "tf_in")
  local encoded = tf.layers.positional_encoding(graph, input, {
    input_shape = { 4, 8 }, name = "pos",
  })
  local block = tf.layers.transformer_block(graph, encoded.output, {
    input_shape = { 4, 8 }, num_heads = 2, key_dim = 4, ff_dim = 16,
    name = "block",
  })
  assert_true(#block.variables > 0)
  local loss = tf.reduce_sum(graph, block.output, { 0, 1, 2 }, "tf_loss",
    "float32")
  local refs = {}
  for index, variable in ipairs(block.variables) do refs[index] = variable.ref end
  local gradients = tf.gradients(graph, { loss }, refs)
  assert_eq(#gradients, #refs)
  local session = tf.Session.new(graph)
  session:run({}, {}, block.initializers)
  local values = {}
  for i = 1, 4 do values[i] = {}; for j = 1, 8 do values[i][j] = i / 4 end end
  local result = session:run({ [input] = { values } },
    { block.output, gradients[1] })[1]
  assert_eq(table.concat(result:shape(), ","), "1,4,8")
  session:close()
end)

test("upsampling3d repeats voxels and preserves shapes", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 2, 1 }, "up3_in")
  local layer = tf.layers.upsampling3d(graph, input, 2, {
    input_shape = { 2, 2, 2, 1 }, name = "up3",
  })
  local session = tf.Session.new(graph)
  local values = { { { { { 1 } }, { { 2 } } }, { { { 3 } }, { { 4 } } } },
    { { { { 5 } }, { { 6 } } }, { { { 7 } }, { { 8 } } } } }
  local output = session:run({ [input] = values }, { layer.output })[1]
  assert_eq(table.concat(output:shape(), ","), "1,4,4,4,1")
  local flat = output:totable()
  assert_eq(flat[1][1][1][1][1], 1)
  assert_eq(flat[1][1][1][2][1], 1)
  assert_eq(flat[1][2][1][1][1], 1)
  session:close()
end)
