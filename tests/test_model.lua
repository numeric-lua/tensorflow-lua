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
  assert_true(metrics.accuracy >= 0 and metrics.accuracy <= 1)
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

test("counts model parameters and renders a summary", function()
  math.randomseed(17)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 3 })
  model:add({ units = 1 })
  assert_eq(model:count_params(), 2 * 3 + 3 + 3 * 1 + 1)
  local summary = model:summary()
  assert_true(summary:find("Total params: 13", 1, true) ~= nil)
  model:close()
end)

test("supports validation_split in fit", function()
  math.randomseed(18)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local history = model:fit({ { 0 }, { 1 }, { 2 }, { 3 }, { 4 } },
    { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } },
    { epochs = 2, validation_split = 0.4 })
  assert_eq(#history.loss, 2)
  assert_eq(#history.val_loss, 2)
  model:close()
end)

test("predicts class indices", function()
  math.randomseed(19)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce",
    optimizer = { type = "adam", learning_rate = 0.05 } })
  local x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
  local y = { 0, 1, 1, 0 }
  model:fit(x, y, { epochs = 100, batch_size = 4 })
  local classes = model:predict_classes(x)
  assert_eq(table.concat(classes:shape(), ","), "4")
  assert_eq(classes:dtype(), "int64")
  local values = classes:totable()
  assert_true(values[1] == 0 or values[1] == 1)
  model:close()
end)

test("trains sparse labels with class weights", function()
  math.randomseed(20)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce",
    optimizer = { type = "adam", learning_rate = 0.05 },
    class_weight = { [0] = 1, [1] = 2 } })
  local history = model:fit({ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
    { 0, 1, 1, 0 }, { epochs = 50 })
  assert_eq(#history.loss, 50)
  assert_true(history.loss[50] >= 0)
  model:close()
end)

test("gets and sets model weights", function()
  math.randomseed(21)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } }, { epochs = 5 })
  local weights = model:get_weights()
  assert_eq(#weights, 2)
  local before = model:predict({ { 3 } }):totable()[1][1]
  model:set_weights(weights)
  local after = model:predict({ { 3 } }):totable()[1][1]
  assert_true(math.abs(before - after) < 1e-6)
  model:close()
end)

test("supports Flatten in a Sequential stack", function()
  math.randomseed(22)
  local model = tf.Sequential.new({ input_shape = { 2, 2 } })
  model:add({ type = "flatten" })
  model:add({ units = 3 })
  assert_eq(model:count_params(), 4 * 3 + 3)
  local summary = model:summary()
  assert_true(summary:find("Flatten(4)", 1, true) ~= nil)
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local x = { { { 1, 2 }, { 3, 4 } }, { { 2, 3 }, { 4, 5 } } }
  local y = { { 1, 2, 3 }, { 2, 3, 4 } }
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  local prediction = model:predict(x)
  assert_eq(table.concat(prediction:shape(), ","), "2,3")
  model:close()
end)

test("predicts class probabilities", function()
  math.randomseed(23)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce",
    optimizer = { type = "adam", learning_rate = 0.05 } })
  local x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }
  model:fit(x, { 0, 1, 1, 0 }, { epochs = 50 })
  local probabilities = model:predict_proba(x)
  assert_eq(table.concat(probabilities:shape(), ","), "4,2")
  local values = probabilities:totable()
  assert_true(values[1][1] + values[1][2] > 0.99)
  model:close()
end)

test("trains a Sequential model with an Embedding layer", function()
  math.randomseed(24)
  local model = tf.Sequential.new({ input_shape = { 3 }, input_dtype = "int32" })
  model:add({ type = "embedding", input_dim = 10, output_dim = 4 })
  model:add({ type = "flatten" })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce",
    optimizer = { type = "adam", learning_rate = 0.05 } })
  assert_eq(model:count_params(), 10 * 4 + 12 * 2 + 2)
  local x = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }
  local history = model:fit(x, { 0, 1, 0 }, { epochs = 20 })
  assert_eq(#history.loss, 20)
  local output = model:predict(x)
  assert_eq(table.concat(output:shape(), ","), "3,2")
  model:close()
end)

test("infers int32 input dtype for embedding-first models", function()
  math.randomseed(25)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ type = "embedding", input_dim = 6, output_dim = 3 })
  model:add({ type = "flatten" })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce" })
  local output = model:predict({ { 1, 2 }, { 3, 4 } })
  assert_eq(table.concat(output:shape(), ","), "2,2")
  model:close()
end)

test("resumes training with optimizer and epoch state", function()
  math.randomseed(26)
  local prefix = "/tmp/tf-lua-resume/ckpt"
  os.execute("rm -rf /tmp/tf-lua-resume")
  local x = { { 0 }, { 1 }, { 2 }, { 3 } }
  local y = { { 1 }, { 3 }, { 5 }, { 7 } }

  local function build()
    local model = tf.Sequential.new({ input_shape = { 1 } })
    model:add({ units = 1 })
    model:compile({ optimizer = { type = "adamw", learning_rate = 0.05,
      weight_decay = 0.01 } })
    return model
  end

  local first = build()
  first:fit(x, y, { epochs = 20 })
  local snapshot = first:predict({ { 4 } }):totable()[1][1]
  local slot_bytes = {}
  for i, variable in ipairs(first._optimizer_state.variables) do
    slot_bytes[i] = first._session:run({}, { variable.read })[1]:bytes()
  end
  first:save_training(prefix)
  local epoch = first._epoch
  first:fit(x, y, { epochs = 20 })
  local continued = first:predict({ { 4 } }):totable()[1][1]
  first:close()

  local resumed = build()
  resumed:load_training(prefix)
  assert_eq(resumed._epoch, epoch)
  assert_eq(#resumed._optimizer_state.variables, #slot_bytes)
  for i, variable in ipairs(resumed._optimizer_state.variables) do
    local bytes = resumed._session:run({}, { variable.read })[1]:bytes()
    assert_eq(bytes, slot_bytes[i], "optimizer slot " .. i)
  end
  local restored = resumed:predict({ { 4 } }):totable()[1][1]
  assert_true(math.abs(snapshot - restored) < 1e-5,
    tostring(snapshot) .. " vs " .. tostring(restored))
  resumed:fit(x, y, { epochs = 20 })
  local resumed_continued = resumed:predict({ { 4 } }):totable()[1][1]
  assert_true(math.abs(continued - resumed_continued) < 1e-2,
    tostring(continued) .. " vs " .. tostring(resumed_continued))
  resumed:close()
  os.execute("rm -rf /tmp/tf-lua-resume")
end)

test("uses conditional dropout in Sequential", function()
  math.randomseed(27)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 4, options = { activation = "relu" } })
  model:add({ type = "dropout", rate = 0.5, seed = 2 })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local history = model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 5 })
  assert_eq(#history.loss, 5)
  local first = model:predict({ { 3 } }):totable()[1][1]
  local second = model:predict({ { 3 } }):totable()[1][1]
  assert_eq(first, second)
  local summary = model:summary()
  assert_true(summary:find("Dropout", 1, true) ~= nil)
  model:close()
end)

test("trains a Sequential Conv2D stack", function()
  math.randomseed(29)
  local model = tf.Sequential.new({ input_shape = { 4, 4, 1 } })
  model:add({ type = "conv2d", filters = 2, kernel_size = 3,
    options = { padding = "SAME", activation = "relu" } })
  model:add({ type = "max_pool2d", kernel_size = 2, stride = 2 })
  model:add({ type = "flatten" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 3 * 3 * 1 * 2 + 2 + 8 * 1 + 1)
  local x = {}
  local y = {}
  for sample = 1, 2 do
    x[sample] = {}
    for r = 1, 4 do
      x[sample][r] = {}
      for c = 1, 4 do x[sample][r][c] = { sample + r * c / 10 } end
    end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 3, batch_size = 2 })
  assert_eq(#history.loss, 3)
  local prediction = model:predict(x)
  assert_eq(table.concat(prediction:shape(), ","), "2,1")
  local summary = model:summary()
  assert_true(summary:find("Conv2D", 1, true) ~= nil)
  model:close()
end)

test("trains a Sequential model with BatchNormalization", function()
  math.randomseed(30)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 4, options = { activation = "relu" } })
  model:add({ type = "batchnorm" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.02 } })
  assert_eq(model:count_params(), 2 * 4 + 4 + 4 + 4 + 4 * 1 + 1)
  local history = model:fit({ { 1, 2 }, { 2, 3 }, { 3, 4 } },
    { { 2 }, { 3 }, { 4 } }, { epochs = 5, batch_size = 3 })
  assert_eq(#history.loss, 5)
  local first = model:predict({ { 4, 5 } }):totable()[1][1]
  local second = model:predict({ { 4, 5 } }):totable()[1][1]
  assert_eq(first, second)
  local summary = model:summary()
  assert_true(summary:find("BatchNormalization", 1, true) ~= nil)
  model:close()
end)

test("restores BatchNorm moving statistics", function()
  math.randomseed(31)
  local prefix = "/tmp/tf-lua-bn-resume/ckpt"
  os.execute("rm -rf /tmp/tf-lua-bn-resume")
  local function build()
    local model = tf.Sequential.new({ input_shape = { 2 } })
    model:add({ units = 4 })
    model:add({ type = "batchnorm" })
    model:add({ units = 1 })
    model:compile({ optimizer = { type = "adam", learning_rate = 0.02 } })
    return model
  end
  local first = build()
  first:fit({ { 1, 2 }, { 2, 3 }, { 3, 4 } }, { { 2 }, { 3 }, { 4 } },
    { epochs = 5, batch_size = 3 })
  local before = first:predict({ { 4, 5 } }):totable()[1][1]
  first:save_training(prefix)
  first:close()
  local second = build()
  second:load_training(prefix)
  local after = second:predict({ { 4, 5 } }):totable()[1][1]
  assert_true(math.abs(before - after) < 1e-5,
    tostring(before) .. " vs " .. tostring(after))
  second:close()
  os.execute("rm -rf /tmp/tf-lua-bn-resume")
end)

test("trains sparse labels with label smoothing", function()
  math.randomseed(32)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce", label_smoothing = 0.1,
    optimizer = { type = "adam", learning_rate = 0.05 } })
  local history = model:fit({ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
    { 0, 1, 1, 0 }, { epochs = 20 })
  assert_eq(#history.loss, 20)
  assert_true(history.loss[20] >= 0)
  model:close()
end)

test("trains sparse labels with sample weights", function()
  math.randomseed(33)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 2 })
  model:compile({ loss = "sparse_ce",
    optimizer = { type = "adam", learning_rate = 0.05 } })
  local history = model:fit({ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
    { 0, 1, 1, 0 }, { epochs = 10, sample_weight = { 1, 2, 1, 2 } })
  assert_eq(#history.loss, 10)
  local loss, metrics = model:evaluate({ { 0, 0 }, { 1, 1 } }, { 0, 0 },
    { sample_weight = { 1, 2 } })
  assert_true(loss >= 0)
  assert_true(metrics ~= nil)
  model:close()
end)

test("trains a Conv2D + BatchNorm + Dense stack", function()
  math.randomseed(34)
  local model = tf.Sequential.new({ input_shape = { 4, 4, 1 } })
  model:add({ type = "conv2d", filters = 2, kernel_size = 3,
    options = { padding = "SAME" } })
  model:add({ type = "batchnorm" })
  model:add({ type = "flatten" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 3 * 3 * 1 * 2 + 2 + 4 + 32 * 1 + 1)
  local x, y = {}, {}
  for sample = 1, 2 do
    x[sample] = {}
    for r = 1, 4 do
      x[sample][r] = {}
      for c = 1, 4 do x[sample][r][c] = { sample + r * c / 10 } end
    end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  assert_eq(table.concat(model:predict(x):shape(), ","), "2,1")
  model:close()
end)

test("trains a GlobalAvgPool2D Sequential stack", function()
  math.randomseed(35)
  local model = tf.Sequential.new({ input_shape = { 4, 4, 2 } })
  model:add({ type = "avg_pool2d", kernel_size = 2, stride = 2 })
  model:add({ type = "global_avg_pool2d" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 2 * 1 + 1)
  local x, y = {}, {}
  for sample = 1, 2 do
    x[sample] = {}
    for r = 1, 4 do
      x[sample][r] = {}
      for c = 1, 4 do x[sample][r][c] = { sample + r, c } end
    end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  assert_eq(table.concat(model:predict(x):shape(), ","), "2,1")
  model:close()
end)

test("trains a Sequential Conv1D stack", function()
  math.randomseed(37)
  local model = tf.Sequential.new({ input_shape = { 5, 1 } })
  model:add({ type = "conv1d", filters = 2, kernel_size = 3,
    options = { padding = "SAME" } })
  model:add({ type = "flatten" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 1 * 3 * 1 * 2 + 2 + 10 * 1 + 1)
  local x, y = {}, {}
  for sample = 1, 2 do
    x[sample] = {}
    for s = 1, 5 do x[sample][s] = { sample + s / 10 } end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  assert_eq(table.concat(model:predict(x):shape(), ","), "2,1")
  model:close()
end)

test("reports top-k accuracy", function()
  math.randomseed(38)
  local model = tf.Sequential.new({ input_shape = { 2 } })
  model:add({ units = 3 })
  model:compile({ loss = "sparse_ce", metrics = { "top_2_accuracy" },
    optimizer = { type = "adam", learning_rate = 0.05 } })
  local history = model:fit({ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
    { 0, 1, 2, 1 }, { epochs = 5 })
  assert_eq(#history["top_2_accuracy"], 5)
  assert_true(history["top_2_accuracy"][5] >= 0)
  assert_true(history["top_2_accuracy"][5] <= 1)
  model:close()
end)

test("trains dense losses with sample weights", function()
  math.randomseed(39)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local history = model:fit({ { 0 }, { 1 }, { 2 } }, { { 1 }, { 3 }, { 5 } },
    { epochs = 5, sample_weight = { 1, 2, 3 } })
  assert_eq(#history.loss, 5)
  local loss = model:evaluate({ { 0 }, { 2 } }, { { 1 }, { 5 } },
    { sample_weight = { 1, 3 } })
  assert_true(loss >= 0)
  model:close()
end)

test("trains Conv1D with rank-3 BatchNormalization", function()
  math.randomseed(40)
  local model = tf.Sequential.new({ input_shape = { 5, 1 } })
  model:add({ type = "conv1d", filters = 2, kernel_size = 3,
    options = { padding = "SAME" } })
  model:add({ type = "batchnorm" })
  model:add({ type = "flatten" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "adam", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 1 * 3 * 1 * 2 + 2 + 4 + 10 * 1 + 1)
  local x, y = {}, {}
  for sample = 1, 2 do
    x[sample] = {}
    for s = 1, 5 do x[sample][s] = { sample + s / 10 } end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  assert_eq(table.concat(model:predict(x):shape(), ","), "2,1")
  model:close()
end)

test("trains an activation and global max-pool stack", function()
  math.randomseed(41)
  local model = tf.Sequential.new({ input_shape = { 4, 4, 2 } })
  model:add({ type = "activation", activation = "relu" })
  model:add({ type = "global_max_pool2d" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 2 * 1 + 1)
  local x, y = {}, {}
  for sample = 1, 2 do
    x[sample] = {}
    for r = 1, 4 do
      x[sample][r] = {}
      for c = 1, 4 do x[sample][r][c] = { sample + r, c } end
    end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  assert_eq(table.concat(model:predict(x):shape(), ","), "2,1")
  model:close()
end)

test("feeds a scheduled learning rate", function()
  math.randomseed(42)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.1 } })
  local seen = {}
  local callback = {
    on_epoch_end = function(_, _, current)
      seen[#seen + 1] = current._current_lr
      current._current_lr = current._current_lr * 0.5
    end,
  }
  local history = model:fit({ { 0 }, { 1 } }, { { 1 }, { 3 } },
    { epochs = 3, callbacks = { callback } })
  assert_eq(#history.loss, 3)
  assert_eq(seen[1], 0.1)
  assert_eq(seen[2], 0.05)
  assert_true(math.abs(seen[3] - 0.025) < 1e-12)
  model:close()
end)

test("trains a depthwise Conv2D Sequential stack", function()
  math.randomseed(44)
  local model = tf.Sequential.new({ input_shape = { 4, 4, 2 } })
  model:add({ type = "zero_padding2d", padding = 1,
    options = { name = "pad" } })
  model:add({ type = "depthwise_conv2d", multiplier = 2, kernel_size = 3,
    options = { padding = "SAME", activation = "relu", name = "depthwise" } })
  model:add({ type = "upsampling2d", scale = 2, options = { name = "up" } })
  model:add({ type = "global_max_pool2d", options = { name = "gmp" } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  assert_eq(model:count_params(), 3 * 3 * 2 * 2 + 4 + 4 * 1 + 1)
  local x, y = {}, {}
  for sample = 1, 2 do
    x[sample] = {}
    for r = 1, 4 do
      x[sample][r] = {}
      for c = 1, 4 do x[sample][r][c] = { sample + r / 10, c / 10 } end
    end
    y[sample] = { sample }
  end
  local history = model:fit(x, y, { epochs = 2, batch_size = 2 })
  assert_eq(#history.loss, 2)
  assert_eq(table.concat(model:predict(x):shape(), ","), "2,1")
  local summary = model:summary()
  assert_true(summary:find("DepthwiseConv2D", 1, true) ~= nil)
  model:close()
end)

test("runs reshape and permute in a Sequential model", function()
  local model = tf.Sequential.new({ input_shape = { 6 } })
  model:add({ type = "reshape", shape = { 2, 3 } })
  model:add({ type = "permute", perm = { 0, 2, 1 } })
  model:add({ type = "flatten" })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 } })
  local prediction = model:predict({ { 1, 2, 3, 4, 5, 6 } })
  assert_eq(table.concat(prediction:shape(), ","), "1,1")
  model:close()
end)

test("trains with global gradient clipping", function()
  math.randomseed(45)
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.1,
    clipnorm = 0.001 } })
  local history = model:fit({ { 0 }, { 1 }, { 2 } }, { { 10 }, { 20 }, { 30 } },
    { epochs = 5 })
  assert_eq(#history.loss, 5)
  for _, value in ipairs(history.loss) do
    assert_true(value == value, "loss must be finite")
  end
  model:close()
end)
