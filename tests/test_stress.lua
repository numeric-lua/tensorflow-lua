local tf = require("tensorflow")
local capi = require("tensorflow.capi")

test("stress-releases tensors, graphs, and sessions", function()
  local before = {
    tensors = capi.released.tensors,
    graphs = capi.released.graphs,
    sessions = capi.released.sessions,
  }
  for i = 1, 200 do
    local value = tf.tensor({ i, i + 1, i + 2 })
    value:destroy()
  end
  for _ = 1, 50 do
    local graph = tf.Graph.new()
    local session = tf.Session.new(graph)
    session:close()
    graph:delete()
  end
  for _ = 1, 5 do collectgarbage("collect") end
  assert_true(capi.released.tensors >= before.tensors + 200)
  assert_true(capi.released.graphs >= before.graphs + 50)
  assert_true(capi.released.sessions >= before.sessions + 50)
end)

test("stress-opens and closes checkpoint readers", function()
  local before = capi.released.readers
  for _ = 1, 20 do
    local reader = tf.CheckpointReader.new(
      TEST_ROOT .. "/tests/fixtures/checkpoint/ckpt")
    assert_true(reader:size() > 0)
    reader:close()
  end
  assert_true(capi.released.readers >= before + 20)
end)
