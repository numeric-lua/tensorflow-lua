local root = (arg and arg[0] or "tests/run.lua"):match("^(.*)/tests/") or "."
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

TEST_ROOT = root
local passed, failed = 0, 0
local failures = {}
local current = "?"

function test(name, fn)
  local ok, err = pcall(fn)
  if ok then
    passed = passed + 1
    io.write(".")
  else
    failed = failed + 1
    failures[#failures + 1] = "[" .. current .. "] " .. name .. "\n  " ..
      tostring(err):gsub("\n", "\n  ")
    io.write("F")
  end
  io.flush()
end

function assert_true(value, message)
  if not value then error(message or "expected truthy value", 2) end
end

function assert_eq(actual, expected, message)
  if actual ~= expected then
    error((message or "assert_eq") .. ": expected " .. tostring(expected) ..
      ", got " .. tostring(actual), 2)
  end
end

function assert_error(fn, substring)
  local ok, err = pcall(fn)
  if ok then error("expected an error", 2) end
  if substring and not tostring(err):find(substring, 1, true) then
    error("error " .. tostring(err) .. " does not contain " .. substring, 2)
  end
end

current = "test_tensor.lua"
dofile(root .. "/tests/test_tensor.lua")

current = "test_graph.lua"
dofile(root .. "/tests/test_graph.lua")

current = "test_protobuf.lua"
dofile(root .. "/tests/test_protobuf.lua")

current = "test_eager.lua"
dofile(root .. "/tests/test_eager.lua")

current = "test_ops.lua"
dofile(root .. "/tests/test_ops.lua")

current = "test_checkpoint_reader.lua"
dofile(root .. "/tests/test_checkpoint_reader.lua")

current = "test_model.lua"
dofile(root .. "/tests/test_model.lua")

current = "test_losses.lua"
dofile(root .. "/tests/test_losses.lua")

current = "test_layers.lua"
dofile(root .. "/tests/test_layers.lua")

current = "test_xor.lua"
dofile(root .. "/tests/test_xor.lua")

current = "test_training.lua"
dofile(root .. "/tests/test_training.lua")

io.write("\n")
for _, failure in ipairs(failures) do io.stderr:write(failure .. "\n\n") end
print(string.format("%d passed, %d failed", passed, failed))
if failed > 0 then os.exit(1) end
