local tf = require("tensorflow")

local function varint(value)
  local bytes = {}
  repeat
    local byte = value % 128
    value = math.floor(value / 128)
    if value > 0 then byte = byte + 128 end
    bytes[#bytes + 1] = string.char(byte)
  until value == 0
  return table.concat(bytes)
end

local function field(number, payload)
  return varint(number * 8 + 2) .. varint(#payload) .. payload
end

local function varint_field(number, value)
  return varint(number * 8) .. varint(value)
end

local function attr_type(value)
  return field(5, field(1, "T") .. field(2, varint_field(6, value)))
end

local function attr_dtype(value)
  return field(5, field(1, "dtype") .. field(2, varint_field(6, value)))
end

local function node(name, op, inputs, attr)
  local data = field(1, name) .. field(2, op)
  for _, input in ipairs(inputs or {}) do data = data .. field(3, input) end
  if attr then data = data .. attr end
  return field(1, data)
end

-- GraphDef: z = x + y, all float32 placeholders.
local graph_def = node("x", "Placeholder", nil, attr_dtype(1)) ..
  node("y", "Placeholder", nil, attr_dtype(1)) ..
  node("z", "AddV2", { "x", "y" }, attr_type(1))

test("imports a hand-built GraphDef and runs feeds/fetches", function()
  local graph = tf.Graph.new():import_graph_def(graph_def)
  local session = tf.Session.new(graph)
  local a = tf.tensor({ 1, 2, 3 })
  local b = tf.tensor({ 10, 20, 30 })
  local z = session:run({ x = a, y = b }, "z:0")
  assert_eq(z:dtype(), "float32")
  assert_eq(table.concat(z:shape(), ","), "3")
  assert_eq(z:totable()[1], 11)
  assert_eq(z:totable()[3], 33)
  session:close()
end)

test("supports named fetch maps and graph export", function()
  local graph = tf.Graph.new():import_graph_def(graph_def)
  local session = tf.Session.new(graph)
  local results = session:run({ x = { 1, 1 }, y = { 2, 2 } },
    { sum = "z:0" })
  assert_eq(results.sum:totable()[2], 3)
  assert_true(#graph:graph_def() > 0)
  session:close()
end)

test("reports missing operations and session errors", function()
  local graph = tf.Graph.new():import_graph_def(graph_def)
  assert_error(function() graph:operation("missing") end, "no operation")
  local session = tf.Session.new(graph)
  assert_error(function() session:run({ x = { 1, 2 }, y = { 3, 4, 5 } }, "z:0") end)
  assert_error(function() session:run({ x = { 1 } }, "missing:0") end,
    "no operation")
  session:close()
end)


test("loads a SavedModel, decodes signatures, and runs them", function()
  local session, graph, signatures = tf.load_saved_model(
    TEST_ROOT .. "/tests/fixtures/add_savedmodel")
  assert_true(signatures.serving_default ~= nil)
  assert_eq(signatures.serving_default.inputs.x.name,
    "serving_default_x:0")
  local result = session:run_signature("serving_default",
    { x = { 1, 2 }, y = { 10, 20 } })
  assert_eq(result.sum:totable()[1], 11)
  assert_eq(result.sum:totable()[2], 22)
  assert_error(function() session:run_signature("missing", {}) end,
    "no signature")
  session:close()
end)

test("builds a graph in Lua and runs it", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "bx")
  local y = tf.placeholder(graph, "float32", { 2 }, "by")
  local z = tf.add(graph, x, y, "bz")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { 1, 2 }, [y] = { 10, 20 } },
    { z })
  assert_eq(result[1]:totable()[1], 11)
  assert_eq(result[1]:totable()[2], 22)
  session:close()
end)

test("builds symbolic gradients for a Lua graph", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", nil, "gx")
  local square = graph:new_operation("Square", "gx2")
  square:add_input(x)
  square:set_type("T", "float32")
  local y = square:finish()
  local dx = tf.gradients(graph, { y }, { x })[1]
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = 3 }, { dx })
  assert_eq(result[1]:totable(), 6)
  session:close()
end)

test("trains a variable with gradient descent", function()
  local graph = tf.Graph.new()
  local variable = tf.variable(graph, 5, "v", { dtype = "float32" })
  local square = graph:new_operation("Square", "loss")
  square:add_input(variable.read)
  square:set_type("T", "float32")
  local loss = square:finish()
  local targets = tf.train.gradient_descent(graph, loss, { variable }, 0.1,
    "optimizer")
  local session = tf.Session.new(graph)
  session:run({}, {}, { variable.initializer })
  for _ = 1, 50 do session:run({}, {}, targets) end
  local value = session:run({}, { variable.read })[1]:totable()
  assert_true(math.abs(value) < 1e-3, tostring(value))
  session:close()
end)

test("closes sessions idempotently and rejects use after close", function()
  local graph = tf.Graph.new():import_graph_def(graph_def)
  local session = tf.Session.new(graph)
  session:close()
  session:close()
  assert_error(function() session:run({ x = { 1 }, y = { 2 } }, "z:0") end,
    "closed")
  graph:delete()
end)

test("returns duplicate fetches positionally", function()
  local graph = tf.Graph.new():import_graph_def(graph_def)
  local session = tf.Session.new(graph)
  local results = session:run({ x = { 1 }, y = { 2 } }, { "z:0", "z:0" })
  assert_eq(#results, 2)
  assert_true(results[1] ~= results[2])
  assert_eq(results[1]:totable()[1], 3)
  assert_eq(results[2]:totable()[1], 3)
  session:close()
end)

test("graph delete invalidates operation lookup", function()
  local graph = tf.Graph.new()
  graph:delete()
  assert_error(function() graph:operation("x") end, "deleted")
end)
