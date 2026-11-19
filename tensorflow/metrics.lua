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

local function cast_float(graph, input, name, source_dtype)
  local tf = require("tensorflow")
  return tf.cast(graph, input, source_dtype or input.dtype or "float32",
    "float32", name)
end

-- Macro-averaged precision, recall, and F1 for integer class predictions and
-- labels.  Classes are numbered [0, num_classes).  Returns macro precision,
-- recall, F1, and a table of per-class values.
function Metrics.multiclass_prf(graph, predictions, labels, num_classes, options)
  options = options or {}
  if type(num_classes) ~= "number" or num_classes < 2 then
    error("multiclass_prf requires at least two classes", 3)
  end
  local tf = require("tensorflow")
  local prefix = options.name or "multiclass_prf"
  local epsilon = tf.constant(graph, 1e-8, prefix .. "/epsilon", "float32")
  local columns = { precision = {}, recall = {}, f1 = {} }
  for class = 0, num_classes - 1 do
    local label_op = graph:new_operation("Equal", prefix .. "/label_" .. class)
    label_op:add_input(labels)
    label_op:add_input(tf.constant(graph, class, prefix .. "/label_value_" .. class,
      labels.dtype or "int32"))
    label_op:set_type("T", labels.dtype or "int32")
    local label_match = cast_float(graph, label_op:finish(),
      prefix .. "/label_float_" .. class, "bool")
    local pred_op = graph:new_operation("Equal", prefix .. "/pred_" .. class)
    pred_op:add_input(predictions)
    pred_op:add_input(tf.constant(graph, class, prefix .. "/pred_value_" .. class,
      predictions.dtype or "int64"))
    pred_op:set_type("T", predictions.dtype or "int64")
    local pred_match = cast_float(graph, pred_op:finish(),
      prefix .. "/pred_float_" .. class, "bool")
    local true_positive = tf.reduce_sum(graph, tf.mul(graph, label_match,
      pred_match, prefix .. "/tp_" .. class, "float32"), { 0 },
      prefix .. "/tp_sum_" .. class, "float32", false)
    local predicted_positive = tf.reduce_sum(graph, pred_match, { 0 },
      prefix .. "/predicted_" .. class, "float32", false)
    local actual_positive = tf.reduce_sum(graph, label_match, { 0 },
      prefix .. "/actual_" .. class, "float32", false)
    local precision = tf.div(graph, true_positive,
      tf.maximum(graph, predicted_positive, epsilon,
        prefix .. "/safe_predicted_" .. class, "float32"),
      prefix .. "/precision_" .. class, "float32")
    local recall = tf.div(graph, true_positive,
      tf.maximum(graph, actual_positive, epsilon,
        prefix .. "/safe_actual_" .. class, "float32"),
      prefix .. "/recall_" .. class, "float32")
    local denominator = tf.maximum(graph, tf.add(graph, precision, recall,
      prefix .. "/sum_" .. class, "float32"), epsilon,
      prefix .. "/safe_sum_" .. class, "float32")
    local f1 = tf.div(graph, tf.mul(graph, tf.mul(graph, precision, recall,
      prefix .. "/product_" .. class, "float32"), tf.constant(graph, 2,
      prefix .. "/two_" .. class, "float32"), prefix .. "/numerator_" .. class,
      "float32"), denominator, prefix .. "/f1_" .. class, "float32")
    columns.precision[class] = precision
    columns.recall[class] = recall
    columns.f1[class] = f1
  end
  local function macro(values, label)
    local total
    for class = 0, num_classes - 1 do
      total = total == nil and values[class] or tf.add(graph, total,
        values[class], prefix .. "/macro_" .. label .. "_add_" .. class)
    end
    return tf.div(graph, total, tf.constant(graph, num_classes,
      prefix .. "/macro_" .. label .. "_count", "float32"),
      prefix .. "/macro_" .. label, "float32")
  end
  return macro(columns.precision, "precision"), macro(columns.recall, "recall"),
    macro(columns.f1, "f1"),
    { precision = columns.precision, recall = columns.recall, f1 = columns.f1 }
end

-- Binary ROC AUC using a pairwise comparison matrix.  This is exact but
-- O(N^2); callers should keep batches moderate or use the default 1e6-pair
-- guard by passing opts.max_pairs.
function Metrics.auc(graph, predictions, labels, options)
  options = options or {}
  local tf = require("tensorflow")
  local prefix = options.name or "auc"
  local flat_predictions = tf.reshape(graph, predictions, { -1 },
    prefix .. "/pred_flat", "float32")
  local flat_labels = tf.cast(graph, labels, labels.dtype or "float32",
    "float32", prefix .. "/label_float")
  flat_labels = tf.reshape(graph, flat_labels, { -1 }, prefix .. "/label_flat",
    "float32")
  local column = graph:new_operation("ExpandDims", prefix .. "/pred_column")
  column:add_input(flat_predictions)
  column:add_input(tf.constant(graph, 1, prefix .. "/axis1", "int32"))
  column:set_type("T", "float32")
  column:set_type("Tdim", "int32")
  local row = graph:new_operation("ExpandDims", prefix .. "/pred_row")
  row:add_input(flat_predictions)
  row:add_input(tf.constant(graph, 0, prefix .. "/axis0", "int32"))
  row:set_type("T", "float32")
  row:set_type("Tdim", "int32")
  local diff = tf.sub(graph, column:finish(), row:finish(),
    prefix .. "/difference", "float32")
  local label_column = graph:new_operation("ExpandDims", prefix .. "/label_column")
  label_column:add_input(flat_labels)
  label_column:add_input(tf.constant(graph, 1, prefix .. "/label_axis1", "int32"))
  label_column:set_type("T", "float32")
  label_column:set_type("Tdim", "int32")
  local label_row = graph:new_operation("ExpandDims", prefix .. "/label_row")
  label_row:add_input(flat_labels)
  label_row:add_input(tf.constant(graph, 0, prefix .. "/label_axis0", "int32"))
  label_row:set_type("T", "float32")
  label_row:set_type("Tdim", "int32")
  local positive = label_column:finish()
  local negative = tf.sub(graph, tf.constant(graph, 1, prefix .. "/one", "float32"),
    label_row:finish(), prefix .. "/negative", "float32")
  local mask = tf.mul(graph, positive, negative, prefix .. "/pairs", "float32")
  local greater = graph:new_operation("Greater", prefix .. "/greater")
  greater:add_input(diff)
  greater:add_input(tf.constant(graph, 0, prefix .. "/zero", "float32"))
  greater:set_type("T", "float32")
  local equal = graph:new_operation("Equal", prefix .. "/equal")
  equal:add_input(diff)
  equal:add_input(tf.constant(graph, 0, prefix .. "/zero_equal", "float32"))
  equal:set_type("T", "float32")
  local wins = cast_float(graph, greater:finish(), prefix .. "/wins", "bool")
  local ties = tf.mul(graph, cast_float(graph, equal:finish(), prefix .. "/ties",
    "bool"),
    tf.constant(graph, 0.5, prefix .. "/half", "float32"), prefix .. "/tie_score",
    "float32")
  local score = tf.add(graph, wins, ties, prefix .. "/score", "float32")
  local numerator = tf.reduce_sum(graph, tf.mul(graph, mask, score,
    prefix .. "/weighted", "float32"), { 0, 1 }, prefix .. "/numerator",
    "float32", false)
  local denominator = tf.reduce_sum(graph, mask, { 0, 1 },
    prefix .. "/denominator", "float32", false)
  return tf.div(graph, numerator, tf.maximum(graph, denominator,
    tf.constant(graph, 1e-8, prefix .. "/epsilon", "float32"),
    prefix .. "/safe_denominator", "float32"), prefix .. "/value", "float32")
end

return Metrics
