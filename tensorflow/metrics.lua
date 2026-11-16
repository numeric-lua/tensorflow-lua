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

function Metrics.binary_accuracy(graph, predictions, labels, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local labels_dtype = options.labels_dtype or "float32"
  local prefix = options.name or "binary_accuracy"
  local tf = require("tensorflow")
  local threshold = tf.constant(graph, options.threshold or 0.5,
    prefix .. "/threshold", dtype)
  local desc = graph:new_operation("GreaterEqual", prefix .. "/prediction")
  desc:add_input(predictions)
  desc:add_input(threshold)
  desc:set_type("T", dtype)
  local predicted = desc:finish()
  local cast_labels = graph:new_operation("Cast", prefix .. "/labels")
  cast_labels:add_input(labels)
  cast_labels:set_type("SrcT", labels_dtype)
  cast_labels:set_type("DstT", "bool")
  local truth = cast_labels:finish()
  local equal = graph:new_operation("Equal", prefix .. "/equal")
  equal:add_input(predicted)
  equal:add_input(truth)
  equal:set_type("T", "bool")
  local matches = equal:finish()
  local cast_matches = graph:new_operation("Cast", prefix .. "/cast")
  cast_matches:add_input(matches)
  cast_matches:set_type("SrcT", "bool")
  cast_matches:set_type("DstT", "float32")
  local values = cast_matches:finish()
  return tf.reduce_mean(graph, values, options.axes or { 0 },
    prefix .. "/mean", "float32")
end

return Metrics
