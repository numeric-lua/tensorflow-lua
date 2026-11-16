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

test("writes and restores a native TensorFlow V2 checkpoint", function()
  local prefix = "/tmp/tf-lua-save-v2/ckpt"
  os.execute("rm -rf /tmp/tf-lua-save-v2")
  local graph = tf.Graph.new()
  local v = tf.variable(graph, 4.25, "native_v")
  local session = tf.Session.new(graph)
  session:run({}, {}, { v.initializer })
  local save = tf.save_v2(graph, prefix, { { name = "native_v",
    variable = v } })
  session:run({}, {}, { save })
  session:close()
  local reader = tf.CheckpointReader.new(prefix)
  local names = reader:variables()
  local name
  for _, candidate in ipairs(names) do
    if candidate:find("native_v", 1, true) then name = candidate end
  end
  assert_true(name ~= nil, "native checkpoint variable missing")
  local saved = reader:tensor(name):totable()
  if type(saved) == "table" then saved = saved[1] end
  assert_eq(saved, 4.25)
  reader:close()

  local graph2 = tf.Graph.new()
  local v2 = tf.variable(graph2, 0, "native_v")
  local restore = tf.restore_v2(graph2, prefix,
    { { name = "native_v", variable = v2 } })
  local session2 = tf.Session.new(graph2)
  session2:run({}, {}, restore)
  assert_eq(session2:run({}, { v2.read })[1]:totable(), 4.25)
  session2:close()
  os.execute("rm -rf /tmp/tf-lua-save-v2")
end)

test("trains linear regression with RMSProp", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", nil, "rx")
  local target = tf.placeholder(graph, "float32", nil, "rtarget")
  local w = tf.variable(graph, 0, "rw")
  local b = tf.variable(graph, 0, "rb")
  local prediction = tf.add(graph, tf.mul(graph, x, w.read, "rwx"), b.read,
    "rprediction")
  local loss = tf.reduce_mean(graph,
    tf.square(graph, tf.sub(graph, prediction, target, "rerror"), "rsquared"),
    { 0 }, "rloss")
  local updates, _, state = tf.train.rmsprop(graph, loss, { w, b },
    { learning_rate = 0.05 })
  local session = tf.Session.new(graph)
  local init = { w.initializer, b.initializer }
  for _, value in ipairs(state.initializers) do init[#init + 1] = value end
  session:run({}, {}, init)
  local xs = { 0, 1, 2, 3 }
  local ys = { 1, 3, 5, 7 }
  for _ = 1, 500 do session:run({ [x] = xs, [target] = ys }, {}, updates) end
  local result = session:run({}, { w.read, b.read })
  assert_true(math.abs(result[1]:totable() - 2) < 0.25,
    tostring(result[1]:totable()))
  assert_true(math.abs(result[2]:totable() - 1) < 0.3,
    tostring(result[2]:totable()))
  session:close()
end)

local function linear_problem()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", nil, "lx")
  local target = tf.placeholder(graph, "float32", nil, "ltarget")
  local w = tf.variable(graph, 0, "lw")
  local b = tf.variable(graph, 0, "lb")
  local prediction = tf.add(graph, tf.mul(graph, x, w.read, "lwx"), b.read,
    "lprediction")
  local loss = tf.reduce_mean(graph,
    tf.square(graph, tf.sub(graph, prediction, target, "lerror"), "lsquared"),
    { 0 }, "lloss")
  return graph, x, target, w, b, loss
end

test("trains linear regression with Adagrad", function()
  local graph, x, target, w, b, loss = linear_problem()
  local updates, _, state = tf.train.adagrad(graph, loss, { w, b },
    { learning_rate = 0.5 })
  local session = tf.Session.new(graph)
  local init = { w.initializer, b.initializer }
  for _, value in ipairs(state.initializers) do init[#init + 1] = value end
  session:run({}, {}, init)
  for _ = 1, 500 do
    session:run({ [x] = { 0, 1, 2, 3 }, [target] = { 1, 3, 5, 7 } }, {}, updates)
  end
  local result = session:run({}, { w.read, b.read })
  assert_true(math.abs(result[1]:totable() - 2) < 0.3,
    tostring(result[1]:totable()))
  assert_true(math.abs(result[2]:totable() - 1) < 0.3,
    tostring(result[2]:totable()))
  session:close()
end)

test("trains linear regression with momentum", function()
  local graph, x, target, w, b, loss = linear_problem()
  local updates, _, state = tf.train.momentum(graph, loss, { w, b },
    { learning_rate = 0.05, momentum = 0.9 })
  local session = tf.Session.new(graph)
  local init = { w.initializer, b.initializer }
  for _, value in ipairs(state.initializers) do init[#init + 1] = value end
  session:run({}, {}, init)
  for _ = 1, 500 do
    session:run({ [x] = { 0, 1, 2, 3 }, [target] = { 1, 3, 5, 7 } }, {}, updates)
  end
  local result = session:run({}, { w.read, b.read })
  assert_true(math.abs(result[1]:totable() - 2) < 0.3,
    tostring(result[1]:totable()))
  assert_true(math.abs(result[2]:totable() - 1) < 0.3,
    tostring(result[2]:totable()))
  session:close()
end)

test("rejects corrupt or incomplete checkpoints", function()
  local path = os.tmpname()
  local file = assert(io.open(path, "wb"))
  file:write("not a checkpoint")
  file:close()
  assert_error(function() tf.load_checkpoint(tf.Graph.new(), path, {}) end)
  os.remove(path)

  local graph = tf.Graph.new()
  local v = tf.variable(graph, 0, "missing_v")
  local session = tf.Session.new(graph)
  session:run({}, {}, { v.initializer })
  local good = os.tmpname()
  tf.save_checkpoint(session, good, { { name = "missing_v", variable = v } })
  session:close()
  local graph2 = tf.Graph.new()
  local other = tf.variable(graph2, 0, "other_v")
  assert_error(function()
    tf.load_checkpoint(graph2, good, { { name = "other_v", variable = other } })
  end, "no variable")
  os.remove(good)
end)

test("writes and reads multiple native checkpoint dtypes", function()
  local prefix = "/tmp/tf-lua-multi-v2/ckpt"
  os.execute("rm -rf /tmp/tf-lua-multi-v2")
  local graph = tf.Graph.new()
  local a = tf.variable(graph, 1.5, "a", { dtype = "float32" })
  local b = tf.variable(graph, 7, "b", { dtype = "int32" })
  local session = tf.Session.new(graph)
  session:run({}, {}, { a.initializer, b.initializer })
  session:run({}, {}, { tf.save_v2(graph, prefix, {
    { name = "a", variable = a }, { name = "b", variable = b } }) })
  session:close()
  local reader = tf.CheckpointReader.new(prefix)
  local float_value = reader:tensor("a"):totable()
  local int_value = reader:tensor("b"):totable()
  assert_eq(float_value, 1.5)
  assert_eq(int_value, 7)
  reader:close()
  os.execute("rm -rf /tmp/tf-lua-multi-v2")
end)
