--
-- Graph-mode metric builders.
--
local Metrics = {}

function Metrics.accuracy(graph, predictions, labels, options)
  options = options or {}
  local dtype = options.dtype or "int32"
  local prefix = options.name or "accuracy"
  local equal = graph:new_operation("Equal", prefix .. "/equal")
  equal:add_input(predictions)
  equal:add_input(labels)
  equal:set_type("T", dtype)
  local matches = equal:finish()
  local cast = graph:new_operation("Cast", prefix .. "/cast")
  cast:add_input(matches)
  cast:set_type("SrcT", "bool")
  cast:set_type("DstT", "float32")
  local values = cast:finish()
  return require("tensorflow").reduce_mean(graph, values, options.axes or { 0 },
    prefix .. "/mean", "float32")
end

return Metrics
