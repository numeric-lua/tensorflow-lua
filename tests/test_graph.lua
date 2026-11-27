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

test("imports prefixed GraphDefs into one graph", function()
  local graph = tf.Graph.new()
  graph:import_graph_def(graph_def, { prefix = "left" })
  graph:import_graph_def(graph_def, { prefix = "right" })
  local session = tf.Session.new(graph)
  local result = session:run({
    ["left/x"] = { 1, 2 },
    ["left/y"] = { 3, 4 },
    ["right/x"] = { 10, 20 },
    ["right/y"] = { 5, 6 },
  }, { left = "left/z:0", right = "right/z:0" })
  assert_eq(result.left:totable()[1], 4)
  assert_eq(result.right:totable()[2], 26)
  session:close()
end)

test("imports GraphDefs with device and uniquify options", function()
  local graph = tf.Graph.new()
  graph:import_graph_def(graph_def)
  graph:import_graph_def(graph_def, { uniquify_names = true })
  local names = graph:operations()
  assert_eq(#names, 6)
  assert_true(graph:operation("x_1") ~= nil)

  local deviced = tf.Graph.new():import_graph_def(graph_def,
    { default_device = "/CPU:0" })
  assert_eq(deviced:operation_device("z"), "/CPU:0")
  assert_eq(deviced:operation_device("x"), "/CPU:0")

  local prefixed = tf.Graph.new():import_graph_def(graph_def, { prefix = "p" })
  prefixed:import_graph_def(graph_def, {
    prefix = "p", uniquify_prefix = true,
  })
  assert_eq(#prefixed:operations(), 6)
  assert_true(prefixed:operation("p_1/x") ~= nil)

  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, { default_device = 1 })
  end, "default_device")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, { uniquify_names = "yes" })
  end, "uniquify_names")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, { uniquify_prefix = "yes" })
  end, "uniquify_prefix")
end)

test("remaps GraphDef inputs to existing graph outputs", function()
  local graph = tf.Graph.new()
  local replacement = tf.constant(graph, { 1, 2, 3 }, "replacement")
  graph:import_graph_def(graph_def, {
    input_map = { ["x:0"] = replacement },
  })
  local session = tf.Session.new(graph)
  local result = session:run({ y = { 10, 20, 30 } }, "z:0")
  assert_eq(result:totable()[1], 11)
  assert_eq(result:totable()[3], 33)

  local graph2 = tf.Graph.new()
  tf.constant(graph2, { 4, 5 }, "mapped")
  graph2:import_graph_def(graph_def, {
    input_map = { x = "mapped:0" },
  })
  local session2 = tf.Session.new(graph2)
  local result2 = session2:run({ y = { 6, 7 } }, "z:0")
  assert_eq(result2:totable()[1], 10)
  assert_eq(result2:totable()[2], 12)
  session:close()
  session2:close()
end)

test("returns selected outputs from GraphDef imports", function()
  local graph = tf.Graph.new()
  local returned = graph:import_graph_def(graph_def, {
    return_elements = { "z:0", "z:0" },
  })
  assert_eq(#returned, 2)
  assert_eq(returned[1].name, "z:0")
  assert_eq(returned[1].dtype, "float32")
  local session = tf.Session.new(graph)
  local result = session:run({ x = { 1, 2 }, y = { 10, 20 } }, returned)
  assert_eq(result[1]:totable()[1], 11)
  assert_eq(result[2]:totable()[2], 22)
  session:close()

  local prefixed = tf.Graph.new()
  local outputs = prefixed:import_graph_def(graph_def, {
    prefix = "selected",
    return_elements = { "selected/z:0" },
  })
  assert_eq(outputs[1].name, "selected/z:0")
  local prefixed_session = tf.Session.new(prefixed)
  assert_eq(prefixed_session:run({ ["selected/x"] = { 4 },
    ["selected/y"] = { 5 } }, outputs)[1]:totable()[1], 9)
  prefixed_session:close()

  local mapped = tf.Graph.new()
  tf.constant(mapped, { 3, 4 }, "mapped_input")
  local mapped_outputs = mapped:import_graph_def(graph_def, {
    input_map = { x = "mapped_input:0" },
    return_elements = { "z:0" },
  })
  local mapped_session = tf.Session.new(mapped)
  assert_eq(mapped_session:run({ y = { 1, 2 } }, mapped_outputs)[1]
    :totable()[2], 6)
  mapped_session:close()

  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, { return_elements = 42 })
  end, "return_elements")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, { return_elements = { 1 } })
  end, "must be a string")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, {
      return_elements = { "missing:0" },
    })
  end, "no operation")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, {
      return_elements = { "z:5" },
    })
  end, "no output")
end)

test("refines graph tensor shapes", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { -1, 2 }, "set_shape_x")
  assert_eq(table.concat(graph:shape(x), ","), "-1,2")
  graph:set_shape(x, { 3, 2 })
  assert_eq(table.concat(graph:shape("set_shape_x:0"), ","), "3,2")
  assert_error(function() graph:set_shape(x, { 4, 2 }) end,
    "graph tensor shape update")
  assert_error(function() graph:set_shape(x, { 2.5, 2 }) end, "integers")
  assert_error(function() graph:set_shape(x, "shape") end, "table or nil")
  assert_error(function() graph:set_shape("set_shape_x:1", { 2 }) end,
    "no output")
  assert_error(function() graph:set_shape("missing", { 2 }) end,
    "no operation")
end)

test("reports reference dtypes in operation info", function()
  local graph = tf.Graph.new()
  local variable = tf.variable(graph, { 1, 2 }, "ref_v")
  local ref_info = graph:operation_info(variable.ref)
  assert_eq(ref_info.op_type, "VariableV2")
  assert_eq(ref_info.outputs[1].dtype, "float32")
  local read_info = graph:operation_info(variable.read)
  assert_eq(read_info.op_type, "Identity")
  assert_eq(read_info.inputs[1].dtype, "float32")
  assert_eq(read_info.outputs[1].dtype, "float32")
end)

test("reports consumers and control edges", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "ctx")
  local y = tf.square(graph, x, "cty")
  local first = graph:new_operation("Identity", "cfirst")
  first:add_input(y)
  first:set_type("T", "float32")
  local first_output = first:finish()
  local second = graph:new_operation("Identity", "csecond")
  second:add_input(y)
  second:set_type("T", "float32")
  second:finish()
  local sink = graph:new_operation("Identity", "csink")
  sink:add_input(x)
  sink:add_control_input(first_output)
  sink:set_type("T", "float32")
  sink:finish()

  local consumers = graph:consumers("cty")
  assert_eq(#consumers, 2)
  local names = { consumers[1].name, consumers[2].name }
  table.sort(names)
  assert_eq(names[1], "cfirst")
  assert_eq(names[2], "csecond")
  assert_eq(consumers[1].input, 0)
  assert_eq(#graph:consumers("ctx"), 2)
  assert_eq(table.concat(graph:control_inputs("csink"), ","), "cfirst")
  assert_eq(table.concat(graph:control_outputs("cfirst"), ","), "csink")

  local info = graph:operation_info("cty")
  assert_eq(info.outputs[1].consumers, 2)
  assert_eq(#info.outputs[1].consumer_ops, 2)
  assert_eq(#info.control_inputs, 0)
  local first_info = graph:operation_info("cfirst")
  assert_eq(#first_info.control_inputs, 0)
  assert_eq(table.concat(first_info.control_outputs, ","), "csink")
  assert_error(function() graph:consumers("cty:5") end, "no output")
  assert_error(function() graph:control_inputs("missing") end, "no operation")
end)

test("returns imported operation metadata", function()
  local graph = tf.Graph.new()
  local operations = graph:import_graph_def(graph_def, {
    return_operations = { "x", "z" },
  })
  assert_eq(#operations, 2)
  assert_eq(operations[1].name, "x")
  assert_eq(operations[1].op_type, "Placeholder")
  assert_eq(operations[2].name, "z")
  assert_eq(operations[2].op_type, "AddV2")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, {
      return_operations = { "x" }, return_elements = { "z:0" },
    })
  end, "cannot combine")
  assert_error(function()
    tf.Graph.new():import_graph_def(graph_def, {
      return_operations = { 1 },
    })
  end, "must be a string")
end)

test("rejects invalid GraphDef input mappings", function()
  local graph = tf.Graph.new()
  local replacement = tf.constant(graph, { 1 }, "replacement")
  assert_error(function()
    graph:import_graph_def(graph_def, { input_map = 42 })
  end, "input_map")
  assert_error(function()
    graph:import_graph_def(graph_def, { input_map = { [1] = replacement } })
  end, "keys")
  assert_error(function()
    graph:import_graph_def(graph_def, { input_map = { [""] = replacement } })
  end, "empty")
  assert_error(function()
    graph:import_graph_def(graph_def, { input_map = { x = "nope:0" } })
  end, "no operation")
end)

test("rejects bad GraphDef import options", function()
  local graph = tf.Graph.new()
  graph:import_graph_def(graph_def, { prefix = "dup" })
  assert_error(function()
    graph:import_graph_def(graph_def, { prefix = 42 })
  end, "prefix")
  assert_error(function()
    graph:import_graph_def(graph_def, true)
  end, "options")
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
  local session, _, signatures = tf.load_saved_model(
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

test("exports operation NodeDefs", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "node_x")
  local data = graph:node_def(x)
  assert_true(#data > 0)
  assert_true(data:find("node_x", 1, true) ~= nil)
  assert_error(function() graph:node_def("node_missing") end, "no operation")
end)

test("exports operation OpDefs", function()
  local graph = tf.Graph.new()
  local data = graph:op_def("AddV2")
  assert_true(#data > 0)
  assert_true(data:find("AddV2", 1, true) ~= nil)
  assert_error(function() graph:op_def("NoSuchOperation") end, "OpDef export")
  assert_error(function() graph:op_def("") end, "non-empty")
end)

test("exports graph version metadata", function()
  local graph = tf.Graph.new()
  local data = graph:version_def()
  assert_true(#data > 0)
  assert_eq(#graph:version_def(), #data)
end)

test("decodes graph version metadata", function()
  local graph = tf.Graph.new()
  local info = graph:version_info()
  assert_true(type(info.producer) == "number")
  assert_true(type(info.min_consumer) == "number")
  assert_true(type(info.bad_consumers) == "table")
  assert_eq(#info.bad_consumers, 0)
end)

test("renders a human-readable graph debug string", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "debug_x")
  tf.square(graph, x, "debug_square")
  local data = graph:debug_string()
  assert_true(#data > 0)
  assert_true(data:find("debug_x", 1, true) ~= nil)
  assert_true(data:find("debug_square", 1, true) ~= nil)
end)

test("round-trips a graph through GraphDef", function()
  local graph = tf.Graph.new():import_graph_def(graph_def)
  local data = graph:graph_def()
  local copy = tf.Graph.new():import_graph_def(data)
  local session = tf.Session.new(copy)
  assert_eq(session:run({ x = { 2 }, y = { 8 } }, "z:0"):totable()[1], 10)
  session:close()
end)

test("loads a multi-output SavedModel", function()
  local session = tf.load_saved_model(
    TEST_ROOT .. "/tests/fixtures/multi_savedmodel")
  local outputs = session:run_signature("serving_default",
    { x = { 3, 4 }, y = { 1, 2 } })
  assert_eq(outputs.sum:totable()[1], 4)
  assert_eq(outputs.diff:totable()[2], 2)
  assert_eq(outputs.product:totable()[1], 3)
  session:close()
end)

test("lists graph operations and infers tensor shapes", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "ix")
  local y = tf.placeholder(graph, "float32", { 2 }, "iy")
  local z = tf.add(graph, x, y, "iz")
  local names = graph:operations()
  assert_eq(#names, 3)
  assert_eq(names[1], "ix")
  assert_eq(names[2], "iy")
  assert_eq(names[3], "iz")

  local info = graph:operation_info(z)
  assert_eq(info.name, "iz")
  assert_eq(info.op_type, "AddV2")
  assert_eq(#info.inputs, 2)
  assert_eq(info.inputs[1].name, "ix:0")
  assert_eq(info.inputs[1].dtype, "float32")
  assert_eq(info.inputs[2].name, "iy:0")
  assert_eq(#info.outputs, 1)
  assert_eq(info.outputs[1].name, "iz:0")
  assert_eq(info.outputs[1].dtype, "float32")
  assert_eq(table.concat(info.outputs[1].shape, ","), "2")
  assert_eq(info.outputs[1].consumers, 0)

  local xinfo = graph:operation_info("ix:0")
  assert_eq(xinfo.op_type, "Placeholder")
  assert_eq(#xinfo.inputs, 0)
  assert_eq(xinfo.outputs[1].consumers, 1)
  assert_eq(table.concat(graph:shape("ix"), ","), "2")
  assert_eq(table.concat(graph:shape(x), ","), "2")

  tf.placeholder(graph, "float32", nil, "iunknown")
  assert_eq(graph:shape("iunknown"), nil)
  assert_error(function() graph:operation_info("missing") end, "no operation")
  assert_error(function() graph:shape(42) end, "expected")
  assert_error(function() graph:shape("iz:5") end, "no output")
  local deleted = tf.Graph.new()
  deleted:delete()
  assert_error(function() deleted:operations() end, "deleted")
end)

test("ffi.gc releases graphs and sessions", function()
  local capi = require("tensorflow.capi")
  local graphs, sessions = capi.released.graphs, capi.released.sessions
  do
    local graph = tf.Graph.new()
    local session = tf.Session.new(graph)
    assert_true(graph ~= nil and session ~= nil)
  end
  for _ = 1, 5 do collectgarbage("collect") end
  assert_true(capi.released.graphs > graphs)
  assert_true(capi.released.sessions > sessions)
end)

test("rejects malformed GraphDefs and missing SavedModels", function()
  assert_error(function() tf.Graph.new():import_graph_def("\255") end)
  assert_error(function() tf.load_saved_model("/definitely/missing/model") end)
end)

test("runs a string SavedModel signature", function()
  local session = tf.load_saved_model(
    TEST_ROOT .. "/tests/fixtures/string_savedmodel")
  local outputs = session:run_signature("serving_default",
    { text = { "hello", "lua" } })
  local values = outputs.upper:totable()
  assert_eq(values[1], "HELLO")
  assert_eq(values[2], "LUA")
  session:close()
end)

test("reads graph operation attributes", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2, 3 }, "attr_x")
  local y = tf.placeholder(graph, "float32", { 2, 3 }, "attr_y")
  tf.nn.leaky_relu(graph, x, "attr_act", "float32", 0.25)
  tf.pad(graph, x, { { 1, 1 }, { 1, 1 } }, "REFLECT", "attr_pad")
  tf.concat(graph, { x, y }, 1, "attr_concat", "float32")
  local identity = graph:new_operation("IdentityN", "attr_identity")
  identity:add_inputs({ x, y })
  identity:set_type_list("T", { "float32", "float32" })
  identity:finish()
  tf.squeeze(graph, tf.placeholder(graph, "float32", { 1, 2, 1 },
    "attr_squeeze_x"), { 0, 2 }, "attr_squeeze")
  tf.constant(graph, { 1, 2 }, "attr_const", "int32")

  local shape_info = graph:attribute_info("attr_x", "shape")
  assert_eq(shape_info.operation, "attr_x")
  assert_eq(shape_info.kind, "shape")
  assert_true(not shape_info.is_list)
  assert_eq(table.concat(graph:attribute(x, "shape"), ","), "2,3")
  assert_eq(graph:attribute("attr_x:0", "dtype"), "float32")
  assert_eq(graph:attribute("attr_act", "alpha"), 0.25)
  assert_eq(graph:attribute("attr_pad", "mode"), "REFLECT")
  assert_eq(graph:attribute("attr_concat", "N"), 2)
  assert_eq(graph:attribute("attr_concat", "Tidx"), "int32")
  assert_eq(table.concat(graph:attribute("attr_squeeze", "squeeze_dims"),
    ","), "0,2")
  local dtypes = graph:attribute("attr_identity", "T")
  assert_eq(#dtypes, 2)
  assert_eq(dtypes[1], "float32")
  assert_eq(dtypes[2], "float32")
  local types_info = graph:attribute_info("attr_identity", "T")
  assert_true(types_info.is_list)
  assert_eq(types_info.kind, "type")
  assert_eq(types_info.list_size, 2)
  assert_eq(graph:attribute("attr_const", "dtype"), "int32")

  local value_info = graph:attribute_info("attr_const", "value")
  assert_eq(value_info.kind, "tensor")
  local value = graph:attribute("attr_const", "value")
  assert_eq(value:dtype(), "int32")
  assert_eq(table.concat(value:totable(), ","), "1,2")
  value:destroy()
  local proto = graph:attribute_proto("attr_pad", "mode")
  assert_true(#proto > 0)
  assert_true(proto:find("REFLECT", 1, true) ~= nil)
  assert_true(#graph:attribute_proto("attr_const", "value") > 0)
  assert_error(function() graph:attribute_proto("attr_x", "") end,
    "non-empty")
  assert_error(function() graph:attribute_proto("attr_x", "missing") end,
    "attribute proto")
  assert_error(function() graph:attribute("attr_x", "missing") end,
    "attribute")
  assert_error(function() graph:attribute_info("attr_x", "") end,
    "non-empty")
  assert_error(function() graph:attribute("attr_missing", "dtype") end,
    "no operation")
end)

test("lists graph operation attribute names", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "names_x")
  tf.nn.leaky_relu(graph, x, "names_act", "float32", 0.1)
  local names = graph:attribute_names("names_act")
  table.sort(names)
  assert_eq(table.concat(names, ","), "T,alpha")
  local placeholder_names = graph:attribute_names(x)
  table.sort(placeholder_names)
  assert_eq(table.concat(placeholder_names, ","), "dtype,shape")
  assert_error(function() graph:attribute_names("names_missing") end,
    "no operation")
end)

test("reads all graph operation attributes", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "attrs_x")
  tf.nn.leaky_relu(graph, x, "attrs_act", "float32", 0.25)
  tf.constant(graph, { 1, 2 }, "attrs_const", "int32")

  local function find(attributes, name)
    for _, attribute in ipairs(attributes) do
      if attribute.name == name then return attribute end
    end
  end

  local placeholder = graph:attributes(x)
  local dtype = find(placeholder, "dtype")
  assert_eq(dtype.kind, "type")
  assert_eq(dtype.value, "float32")
  local shape = find(placeholder, "shape")
  assert_eq(shape.kind, "shape")
  assert_eq(table.concat(shape.value, ","), "2")

  local activation = find(graph:attributes("attrs_act"), "alpha")
  assert_eq(activation.kind, "float")
  assert_eq(activation.value, 0.25)

  local constant = find(graph:attributes("attrs_const"), "value")
  assert_eq(constant.kind, "tensor")
  assert_eq(constant.value:dtype(), "int32")
  assert_eq(table.concat(constant.value:totable(), ","), "1,2")
  constant.value:destroy()
  assert_error(function() graph:attributes("attrs_missing") end,
    "no operation")
end)

test("reports operation devices and flattened inputs", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "device_x")
  local device = graph:new_operation("Identity", "device_op")
  device:set_device("/CPU:0")
  device:add_input(x)
  device:set_type("T", "float32")
  device:finish()
  tf.stack(graph, { x, x, x }, 0, "device_stack")
  assert_eq(graph:operation_device("device_op"), "/CPU:0")
  assert_eq(graph:operation_device("device_x"), nil)
  local inputs = graph:all_inputs("device_stack")
  assert_eq(#inputs, 3)
  assert_eq(inputs[1].name, "device_x:0")
  assert_eq(inputs[1].index, 0)
  assert_eq(inputs[1].dtype, "float32")
  assert_eq(inputs[1].input, 0)
  assert_eq(inputs[3].name, "device_x:0")
  assert_error(function() graph:operation_device("device_missing") end,
    "no operation")
  assert_error(function() graph:all_inputs("device_missing") end,
    "no operation")
end)

test("reports operation input and output list lengths", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "list_x")
  tf.stack(graph, { x, x, x }, 0, "list_stack")
  tf.split(graph, x, 2, 0, "list_split")
  tf.unstack(graph, tf.reshape(graph, x, { 1, 2 }, "list_reshape"), 1, 0,
    "list_unstack")
  assert_eq(graph:input_list_length("list_stack", "values"), 3)
  assert_eq(graph:output_list_length("list_split", "output"), 2)
  assert_eq(graph:output_list_length("list_unstack", "output"), 1)
  assert_error(function()
    graph:input_list_length("list_stack", "missing")
  end, "input list length")
  assert_error(function()
    graph:output_list_length("list_split", "missing")
  end, "output list length")
  assert_error(function() graph:input_list_length("list_stack", "") end,
    "non-empty")
  assert_error(function() graph:input_list_length("list_missing", "values") end,
    "no operation")
end)
