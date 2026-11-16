local tf = require("tensorflow")

test("creates and converts numeric tensors", function()
  local t = tf.tensor({ { 1, 2 }, { 3, 4 } })
  assert_eq(t:dtype(), "float32")
  assert_eq(t:dim(), 2)
  assert_eq(t:numel(), 4)
  assert_eq(table.concat(t:shape(), ","), "2,2")
  local values = t:totable()
  assert_eq(values[2][1], 3)
end)

test("supports explicit dtypes, scalars, and bools", function()
  assert_eq(tf.tensor({ 1, 2 }, { dtype = "int64" }):totable()[2], 2)
  local scalar = tf.tensor(3.5)
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:totable(), 3.5)
  local flags = tf.tensor({ true, false, true }, { dtype = "bool" })
  assert_eq(flags:totable()[2], false)
end)

test("rejects malformed tensors", function()
  assert_error(function() tf.tensor({ { 1 }, { 2, 3 } }) end, "rectangular")
  assert_error(function() tf.tensor({ 1 }, { dtype = "variant" }) end,
    "unsupported")
end)

test("bool tensors treat numeric zero as false", function()
  local values = tf.tensor({ 0, 1, false, true }, { dtype = "bool" }):totable()
  assert_eq(values[1], false)
  assert_eq(values[2], true)
  assert_eq(values[3], false)
  assert_eq(values[4], true)
end)

test("rejects maps, holes, and invalid bool values", function()
  assert_error(function() tf.tensor({ a = 1, b = 2 }) end, "contiguous")
  assert_error(function() tf.tensor({ 1, nil, 3 }) end, "nil holes")
  assert_error(function() tf.tensor({ 0, "x" }) end, "numbers")
  assert_error(function() tf.tensor({ 0, 2 }, { dtype = "bool" }) end,
    "bool tensors")
end)

test("int64 conversion and explicit destroy", function()
  local value = tf.tensor({ 1, 2 }, { dtype = "int64" })
  assert_eq(type(value:totable()[1]), "number")
  value:destroy()
  assert_error(function() value:handle() end, "destroyed")
end)

test("ffi.gc releases tensor handles", function()
  local capi = require("tensorflow.capi")
  local before = capi.released.tensors
  do
    local value = tf.tensor({ 1, 2, 3 })
    assert_eq(value:numel(), 3)
  end
  for _ = 1, 5 do collectgarbage("collect") end
  assert_true(capi.released.tensors > before)
end)

test("round-trips every supported dtype", function()
  local cases = {
    { "float32", { 1.5, 2.5 } }, { "float64", { 1.5, 2.5 } },
    { "int8", { -1, 2 } }, { "int16", { -1, 2 } },
    { "int32", { -1, 2 } }, { "int64", { -1, 2 } },
    { "uint8", { 1, 2 } }, { "uint16", { 1, 2 } },
    { "uint32", { 1, 2 } }, { "uint64", { 1, 2 } },
    { "bool", { true, false } },
    { "qint8", { -1, 2 } }, { "quint8", { 1, 2 } },
    { "qint16", { -1, 2 } }, { "quint16", { 1, 2 } },
    { "qint32", { -1, 2 } },
  }
  for _, case in ipairs(cases) do
    local value = tf.tensor(case[2], { dtype = case[1] })
    assert_eq(value:dtype(), case[1])
    local result = value:totable()
    if case[1] == "bool" then assert_eq(result[1], true)
    else assert_eq(result[1], case[2][1]) end
  end
end)

test("creates and reads TF_STRING tensors", function()
  local graph = tf.Graph.new()
  local constant = tf.constant(graph, { "alpha", "b" }, "strings", "string")
  local identity = graph:new_operation("Identity", "strings_out")
  identity:add_input(constant)
  identity:set_type("T", "string")
  local output = identity:finish()
  local session = tf.Session.new(graph)
  local values = session:run({}, { output })[1]:totable()
  assert_eq(values[1], "alpha")
  assert_eq(values[2], "b")
  session:close()
end)

test("round-trips complex tensors", function()
  local values = tf.tensor({ tf.complex(1, 2), tf.complex(-3, 0.5) },
    { dtype = "complex64" })
  local result = values:totable()
  assert_eq(result[1].real, 1)
  assert_eq(result[1].imag, 2)
  assert_eq(result[2].real, -3)
  assert_eq(result[2].imag, 0.5)
  assert_eq(values:dtype(), "complex64")
  assert_true(tostring(result[1]):find("1") ~= nil)
end)

test("round-trips float16 and bfloat16 tensors", function()
  local half = tf.tensor({ 1.5, -2.25, 0.5 }, { dtype = "float16" }):totable()
  assert_true(math.abs(half[1] - 1.5) < 0.01)
  assert_true(math.abs(half[2] + 2.25) < 0.01)
  assert_true(math.abs(half[3] - 0.5) < 0.01)
  local bfloat = tf.tensor({ 1.5, -2.25 }, { dtype = "bfloat16" }):totable()
  assert_true(math.abs(bfloat[1] - 1.5) < 0.02)
  assert_true(math.abs(bfloat[2] + 2.25) < 0.02)
end)
