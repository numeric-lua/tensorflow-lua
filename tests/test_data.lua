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
