local tf = require("tensorflow")

test("batches datasets without shuffling", function()
  local dataset = tf.data.Dataset.new({ 1, 2, 3, 4, 5 }, { 10, 20, 30, 40,
    50 }):batch(2)
  local batches = {}
  for batch in dataset:each() do batches[#batches + 1] = batch end
  assert_eq(#batches, 3)
  assert_eq(#batches[1].x, 2)
  assert_eq(batches[1].x[2], 2)
  assert_eq(batches[2].y[1], 30)
  assert_eq(#batches[3].x, 1)
end)

test("shuffles deterministically with a seed", function()
  local first = tf.data.Dataset.new({ 1, 2, 3, 4 }):shuffle(5):batch(4)
  local second = tf.data.Dataset.new({ 1, 2, 3, 4 }):shuffle(5):batch(4)
  local first_batch = first:each()()
  local second_batch = second:each()()
  local a = first_batch.x
  local b = second_batch.x
  assert_eq(table.concat(a, ","), table.concat(b, ","))
end)

test("validates dataset arguments", function()
  assert_error(function() tf.data.Dataset.new({ 1 }, { 1, 2 }) end, "length")
  local dataset = tf.data.Dataset.new({ 1 })
  assert_error(function() dataset:batch(0) end, "positive")
end)

test("splits datasets into train and validation partitions", function()
  local train, validation = tf.data.Dataset.new({ 1, 2, 3, 4 },
    { 10, 20, 30, 40 }):split(0.75)
  assert_eq(train:size(), 3)
  assert_eq(validation:size(), 1)
  local batch = train:each()()
  assert_eq(batch.y[3], 30)
end)

test("repeats and truncates dataset iteration", function()
  local dataset = tf.data.Dataset.new({ 1, 2, 3 }):repeat_times(2):take(4):batch(2)
  local batches = {}
  for batch in dataset:each() do batches[#batches + 1] = batch end
  assert_eq(#batches, 2)
  assert_eq(#batches[2].x, 2)
  assert_eq(batches[2].x[2], 1)
end)

test("validates repeat and take counts", function()
  local dataset = tf.data.Dataset.new({ 1 })
  assert_error(function() dataset:repeat_times(0) end, "positive")
  assert_error(function() dataset:take(-1) end, "non-negative")
end)

test("drops partial batches when requested", function()
  local dataset = tf.data.Dataset.new({ 1, 2, 3, 4, 5 }):batch(2,
    { drop_last = true })
  local batches = {}
  for batch in dataset:each() do batches[#batches + 1] = batch end
  assert_eq(#batches, 2)
  assert_eq(#batches[2].x, 2)
end)

test("maps and filters dataset samples", function()
  local dataset = tf.data.Dataset.new({ 1, 2, 3, 4 }, { 10, 20, 30, 40 })
  local mapped = dataset:map(function(x, y) return x * 2, y + 1 end)
  local batch = mapped:each()()
  assert_eq(batch.x[1], 2)
  assert_eq(batch.y[1], 11)
  local filtered = dataset:filter(function(x) return x % 2 == 0 end)
  assert_eq(filtered:size(), 2)
  local filtered_batch = filtered:each()()
  assert_eq(filtered_batch.x[1], 2)
end)

test("tensor-backed Dataset batches and feeds fit_dataset", function()
  local x = tf.tensor({ { 1 }, { 2 }, { 3 }, { 4 } })
  local y = tf.tensor({ { 2 }, { 4 }, { 6 }, { 8 } })
  local dataset = tf.data.Dataset.from_tensor(x, y):batch(2):shuffle(7)
  assert_eq(dataset:size(), 4)
  local seen = 0
  local iterator = dataset:each()
  while true do
    local batch = iterator()
    if batch == nil then break end
    assert_eq(table.concat(batch.x:shape(), ","), "2,1")
    seen = seen + 1
    batch.x:destroy(); batch.y:destroy()
  end
  assert_eq(seen, 2)

  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.001 }, loss = "mse" })
  local clean = tf.data.Dataset.from_tensor(x, y):batch(2)
  local history = model:fit_dataset(clean, { epochs = 3 })
  assert_eq(#history.loss, 3)
  local loss = model:evaluate_dataset(clean)
  assert_true(loss >= 0)
  local prediction = model:predict_dataset(clean)
  assert_eq(prediction:shape()[1], 4)
  prediction:destroy()
  x:destroy(); y:destroy(); model:close()
end)
