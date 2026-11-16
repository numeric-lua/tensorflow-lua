--
-- Small, dependency-free benchmark harness.
--
local root = (arg and arg[0] or "bench/benchmark.lua"):match("^(.*)/bench/") or "."
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

local tf = require("tensorflow")

local function bench(name, iterations, fn)
  collectgarbage("collect")
  local start = os.clock()
  for i = 1, iterations do fn(i) end
  local elapsed = os.clock() - start
  print(string.format("%-24s %8d iterations %8.3f ms  %10.1f ops/s",
    name, iterations, elapsed * 1000, iterations / elapsed))
end

bench("tensor create/totable", 2000, function(i)
  local tensor = tf.tensor({ i, i + 1, i + 2 })
  local _ = tensor:totable()
end)

local graph = tf.Graph.new()
local x = tf.placeholder(graph, "float32", { 4 }, "x")
local y = tf.placeholder(graph, "float32", { 4 }, "y")
local z = tf.add(graph, x, y, "z")
local session = tf.Session.new(graph)
local a = tf.tensor({ 1, 2, 3, 4 })
local b = tf.tensor({ 4, 3, 2, 1 })
bench("session add", 1000, function()
  local _ = session:run({ [x] = a, [y] = b }, { z })[1]:totable()
end)
session:close()

print("version", tf.version(), "binding", tf.VERSION)
