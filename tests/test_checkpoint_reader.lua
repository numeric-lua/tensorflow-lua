local tf = require("tensorflow")

test("reads variables from an official TensorFlow checkpoint", function()
  local reader = tf.CheckpointReader.new(
    TEST_ROOT .. "/tests/fixtures/checkpoint/ckpt")
  assert_eq(reader:size(), 3)
  local names = reader:variables()
  local vname, wname
  for _, name in ipairs(names) do
    if name:find("v/", 1, true) then vname = name end
    if name:find("w/", 1, true) then wname = name end
  end
  assert_true(vname ~= nil and wname ~= nil)
  assert_true(reader:has(vname))
  local v = reader:tensor(vname)
  assert_eq(v:dtype(), "float32")
  assert_eq(table.concat(v:shape(), ","), "3")
  assert_eq(reader:dtype(vname), "float32")
  assert_eq(table.concat(reader:shape(vname), ","), "3")
  local info = reader:info(vname)
  assert_eq(info.name, vname)
  assert_eq(info.dtype, "float32")
  assert_eq(table.concat(info.shape, ","), "3")
  local values = v:totable()
  assert_eq(values[1], 1)
  assert_eq(values[3], 3)
  assert_error(function() reader:tensor("missing") end)
  assert_error(function() reader:dtype("missing") end, "no variable")
  assert_error(function() reader:shape("missing") end, "no variable")
  assert_error(function() reader:info("missing") end, "no variable")
  assert_error(function() reader:dtype("") end, "non-empty")
  reader:close()
end)

test("reports checkpoint variable dtype and shape metadata", function()
  local reader = tf.CheckpointReader.new(
    TEST_ROOT .. "/tests/fixtures/checkpoint/ckpt")
  local vname
  for _, name in ipairs(reader:variables()) do
    if name:find("v/", 1, true) then vname = name end
  end
  assert_true(vname ~= nil)
  assert_eq(reader:dtype(vname), "float32")
  assert_eq(table.concat(reader:shape(vname), ","), "3")
  local info = reader:info(vname)
  assert_eq(info.name, vname)
  assert_eq(info.dtype, "float32")
  assert_eq(table.concat(info.shape, ","), "3")
  assert_error(function() reader:dtype("missing") end, "no variable")
  assert_error(function() reader:shape("missing") end, "no variable")
  reader:close()
end)
