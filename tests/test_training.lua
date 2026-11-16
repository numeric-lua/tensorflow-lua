local tf = require("tensorflow")

test("trains a linear regression graph end to end", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", nil, "x")
  local target = tf.placeholder(graph, "float32", nil, "target")
  local w = tf.variable(graph, 0, "w")
  local b = tf.variable(graph, 0, "b")
  local prediction = tf.add(graph, tf.mul(graph, x, w.read, "wx"), b.read,
    "prediction")
  local error = tf.sub(graph, prediction, target, "error")
  local squared = tf.square(graph, error, "squared")
  local axis = tf.constant(graph, { 0 }, "axis", "int32")
  local mean = graph:new_operation("Mean", "loss")
  mean:add_input(squared)
  mean:add_input(axis)
  mean:set_type("T", "float32")
  mean:set_type("Tidx", "int32")
  mean:set_bool("keep_dims", false)
  local loss = mean:finish()
  local updates = tf.train.gradient_descent(graph, loss, { w, b }, 0.05,
    "train")
  local session = tf.Session.new(graph)
  session:run({}, {}, { w.initializer, b.initializer })
  local xs = { 0, 1, 2, 3 }
  local ys = { 1, 3, 5, 7 }
  for _ = 1, 300 do session:run({ [x] = xs, [target] = ys }, {}, updates) end
  local result = session:run({}, { w.read, b.read })
  assert_true(math.abs(result[1]:totable() - 2) < 0.15,
    tostring(result[1]:totable()))
  assert_true(math.abs(result[2]:totable() - 1) < 0.2,
    tostring(result[2]:totable()))
  session:close()
end)

test("trains linear regression with Adam", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", nil, "ax")
  local target = tf.placeholder(graph, "float32", nil, "atarget")
  local w = tf.variable(graph, 0, "aw")
  local b = tf.variable(graph, 0, "ab")
  local prediction = tf.add(graph, tf.mul(graph, x, w.read, "awx"), b.read,
    "aprediction")
  local error = tf.sub(graph, prediction, target, "aerror")
  local squared = tf.square(graph, error, "asquared")
  local axis = tf.constant(graph, { 0 }, "aaxis", "int32")
  local mean = graph:new_operation("Mean", "aloss")
  mean:add_input(squared); mean:add_input(axis)
  mean:set_type("T", "float32"); mean:set_type("Tidx", "int32")
  mean:set_bool("keep_dims", false)
  local loss = mean:finish()
  local updates, _, state = tf.train.adam(graph, loss, { w, b },
    { learning_rate = 0.1 })
  local session = tf.Session.new(graph)
  session:run({}, {}, { w.initializer, b.initializer, state.initializers[1],
    state.initializers[2], state.initializers[3], state.initializers[4],
    state.initializers[5], state.initializers[6] })
  local xs = { 0, 1, 2, 3 }
  local ys = { 1, 3, 5, 7 }
  for _ = 1, 250 do session:run({ [x] = xs, [target] = ys }, {}, updates) end
  local result = session:run({}, { w.read, b.read })
  assert_true(math.abs(result[1]:totable() - 2) < 0.2,
    tostring(result[1]:totable()))
  assert_true(math.abs(result[2]:totable() - 1) < 0.25,
    tostring(result[2]:totable()))
  session:close()
end)

test("saves and restores variable values", function()
  local path = os.tmpname()
  local graph = tf.Graph.new()
  local v = tf.variable(graph, 3.5, "ckpt_v")
  local session = tf.Session.new(graph)
  session:run({}, {}, { v.initializer })
  tf.save_checkpoint(session, path, { { name = "ckpt_v", variable = v } })
  session:close()

  local graph2 = tf.Graph.new()
  local v2 = tf.variable(graph2, 0, "ckpt_v")
  local restore = tf.load_checkpoint(graph2, path, { { name = "ckpt_v",
    variable = v2 } })
  local session2 = tf.Session.new(graph2)
  session2:run({}, {}, restore)
  assert_eq(session2:run({}, { v2.read })[1]:totable(), 3.5)
  session2:close()
  os.remove(path)
end)
