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
