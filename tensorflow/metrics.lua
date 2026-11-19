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

local function binary_counts(graph, predictions, labels, options)
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "binary_metric"
  local threshold = tf.constant(graph, options.threshold or 0.5,
    prefix .. "/threshold", dtype)
  local predicted = graph:new_operation("GreaterEqual", prefix .. "/predicted")
  predicted:add_input(predictions)
  predicted:add_input(threshold)
  predicted:set_type("T", dtype)
  local positive = tf.cast(graph, predicted:finish(), "bool", dtype,
    prefix .. "/positive")
  local truth = tf.cast(graph, labels, options.labels_dtype or dtype, "bool",
    prefix .. "/truth")
  local truth_float = tf.cast(graph, truth, "bool", dtype, prefix .. "/truth_float")
  local both = tf.mul(graph, positive, truth_float, prefix .. "/true_positive",
    dtype)
  local true_positive = tf.reduce_sum(graph, both, options.axes or { 0 },
    prefix .. "/tp", dtype, false)
  local predicted_positive = tf.reduce_sum(graph, positive,
    options.axes or { 0 }, prefix .. "/predicted_positive", dtype, false)
  local actual_positive = tf.reduce_sum(graph, truth_float,
    options.axes or { 0 }, prefix .. "/actual_positive", dtype, false)
  local epsilon = tf.constant(graph, 1e-8, prefix .. "/epsilon", dtype)
  return true_positive, predicted_positive, actual_positive, epsilon
end

function Metrics.precision(graph, predictions, labels, options)
  options = options or {}
  local tf = require("tensorflow")
  local tp, predicted, _, epsilon = binary_counts(graph, predictions, labels,
    options)
  local name = options.name or "precision"
  return tf.div(graph, tp, tf.maximum(graph, predicted, epsilon,
    name .. "/safe_predicted", options.dtype or "float32"), name, options.dtype)
end

function Metrics.recall(graph, predictions, labels, options)
  options = options or {}
  local tf = require("tensorflow")
  local tp, _, actual, epsilon = binary_counts(graph, predictions, labels,
    options)
  local name = options.name or "recall"
  return tf.div(graph, tp, tf.maximum(graph, actual, epsilon,
    name .. "/safe_actual", options.dtype or "float32"), name,
    options.dtype or "float32")
end

function Metrics.f1(graph, predictions, labels, options)
  options = options or {}
  local tf = require("tensorflow")
  local precision = Metrics.precision(graph, predictions, labels,
    { dtype = options.dtype, threshold = options.threshold,
      labels_dtype = options.labels_dtype, axes = options.axes,
      name = (options.name or "f1") .. "_precision" })
  local recall = Metrics.recall(graph, predictions, labels,
    { dtype = options.dtype, threshold = options.threshold,
      labels_dtype = options.labels_dtype, axes = options.axes,
      name = (options.name or "f1") .. "_recall" })
  local tf = require("tensorflow")
  local name = options.name or "f1"
  local sum = tf.add(graph, precision, recall, name .. "/sum", options.dtype)
  local epsilon = tf.constant(graph, 1e-8, name .. "/epsilon", options.dtype)
  local denominator = tf.maximum(graph, sum, epsilon, name .. "/safe_sum",
    options.dtype)
  local product = tf.mul(graph, precision, recall, name .. "/product",
    options.dtype)
  return tf.div(graph, tf.mul(graph, product, tf.constant(graph, 2,
    name .. "/two", options.dtype), name .. "/numerator", options.dtype),
    denominator, name, options.dtype)
end

return Metrics
