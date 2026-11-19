--
-- Common graph-mode loss builders.
--
local Losses = {}

local function tf()
  return require("tensorflow")
end

function Losses.mean_squared_error(graph, predictions, targets, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local error = tf().sub(graph, predictions, targets,
    (options.name or "mse") .. "/error", dtype)
  local squared = tf().square(graph, error,
    (options.name or "mse") .. "/square", dtype)
  return tf().reduce_mean(graph, squared, options.axes or { 0 },
    (options.name or "mse") .. "/mean", dtype, options.keep_dims)
end

function Losses.softmax_cross_entropy(graph, logits, labels, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "softmax_cross_entropy"
  local desc = graph:new_operation("SoftmaxCrossEntropyWithLogits",
    prefix .. "/loss")
  desc:add_input(logits)
  desc:add_input(labels)
  desc:set_type("T", dtype)
  local loss = desc:finish()
  if options.per_example then return loss end
  if options.class_weights ~= nil then
    local weights = options.class_weights
    if type(weights) ~= "table" then
      error("softmax_cross_entropy class_weights must be a table", 3)
    end
    local classes = 0
    for key in pairs(weights) do
      if type(key) ~= "number" then
        error("softmax_cross_entropy class_weights keys must be numeric", 3)
      end
      classes = math.max(classes, key + 1)
    end
    local values = {}
    for index = 0, classes - 1 do values[#values + 1] = weights[index] or 1 end
    local vector = tf().constant(graph, values, prefix .. "/class_weights",
      dtype)
    local per_class = tf().mul(graph, labels, vector,
      prefix .. "/class_weighted_labels", dtype)
    local per_example = tf().reduce_sum(graph, per_class, { 1 },
      prefix .. "/class_weight", dtype, false)
    local weighted = tf().mul(graph, loss, per_example,
      prefix .. "/weighted", dtype)
    local total = tf().reduce_sum(graph, per_example, { 0 },
      prefix .. "/weight_total", dtype, false)
    local epsilon = tf().constant(graph, 1e-8, prefix .. "/epsilon", dtype)
    return tf().div(graph, tf().reduce_sum(graph, weighted, options.axes or { 0 },
      prefix .. "/weighted_sum", dtype, options.keep_dims), tf().maximum(graph,
      total, epsilon, prefix .. "/safe_total", dtype), prefix .. "/mean", dtype)
  end
  return tf().reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.huber(graph, predictions, targets, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "huber"
  local tf = require("tensorflow")
  local delta = options.delta or 1
  local error = tf.sub(graph, predictions, targets, prefix .. "/error", dtype)
  local absolute = tf.abs(graph, error, prefix .. "/abs", dtype)
  local delta_tensor = tf.constant(graph, delta, prefix .. "/delta", dtype)
  local clipped = tf.minimum(graph, absolute, delta_tensor,
    prefix .. "/clipped", dtype)
  local quadratic = tf.mul(graph, tf.square(graph, clipped,
    prefix .. "/square", dtype), tf.constant(graph, 0.5, prefix .. "/half",
    dtype), prefix .. "/quadratic", dtype)
  local excess = tf.nn.relu(graph, tf.sub(graph, absolute, delta_tensor,
    prefix .. "/excess", dtype), prefix .. "/excess_relu", dtype)
  local linear = tf.mul(graph, excess, delta_tensor, prefix .. "/linear",
    dtype)
  local loss = tf.add(graph, quadratic, linear, prefix .. "/loss", dtype)
  return tf.reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.hinge(graph, predictions, targets, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "hinge"
  local tf = require("tensorflow")
  local margin = tf.constant(graph, options.margin or 1, prefix .. "/margin",
    dtype)
  local product = tf.mul(graph, predictions, targets, prefix .. "/product",
    dtype)
  local loss = tf.nn.relu(graph, tf.sub(graph, margin, product,
    prefix .. "/violation", dtype), prefix .. "/loss", dtype)
  return tf.reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.categorical_hinge(graph, predictions, targets, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "categorical_hinge"
  local tf = require("tensorflow")
  local product = tf.mul(graph, predictions, targets, prefix .. "/product",
    dtype)
  local correct = tf.reduce_sum(graph, product, { 1 }, prefix .. "/correct",
    dtype, false)
  local inverted = tf.sub(graph, tf.constant(graph, 1, prefix .. "/one", dtype),
    targets, prefix .. "/inverted", dtype)
  local wrong = tf.mul(graph, predictions, inverted, prefix .. "/wrong", dtype)
  local max_wrong = tf.reduce_max(graph, wrong, { 1 }, prefix .. "/max_wrong",
    dtype, false)
  local margin_correct = tf.sub(graph, tf.constant(graph, 1,
    prefix .. "/margin", dtype), correct, prefix .. "/margin_correct", dtype)
  local violation = tf.add(graph, margin_correct, max_wrong,
    prefix .. "/violation", dtype)
  local loss = tf.nn.relu(graph, violation, prefix .. "/loss", dtype)
  if options.per_example then return loss end
  return tf.reduce_mean(graph, loss, { 0 }, prefix .. "/mean", dtype,
    options.keep_dims)
end

function Losses.mean_absolute_error(graph, predictions, targets, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "mae"
  local tensorflow = require("tensorflow")
  local error = tensorflow.sub(graph, predictions, targets, prefix .. "/error",
    dtype)
  local absolute = tensorflow.abs(graph, error, prefix .. "/abs", dtype)
  return tensorflow.reduce_mean(graph, absolute, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.binary_cross_entropy(graph, logits, labels, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "binary_cross_entropy"
  local tensorflow = require("tensorflow")
  local negative = tensorflow.mul(graph, logits, tensorflow.constant(graph, -1,
    prefix .. "/negative", dtype), prefix .. "/negate", dtype)
  local relu = tensorflow.nn.relu(graph, logits, prefix .. "/relu", dtype)
  local product = tensorflow.mul(graph, logits, labels, prefix .. "/product",
    dtype)
  local softplus = tensorflow.nn.softplus(graph, negative,
    prefix .. "/softplus", dtype)
  local loss = tensorflow.add(graph, tensorflow.sub(graph, relu, product,
    prefix .. "/sub", dtype), softplus, prefix .. "/loss", dtype)
  return tensorflow.reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.sparse_softmax_cross_entropy(graph, logits, labels, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local labels_dtype = options.labels_dtype or "int32"
  local prefix = options.name or "sparse_softmax_cross_entropy"
  local desc = graph:new_operation("SparseSoftmaxCrossEntropyWithLogits",
    prefix .. "/loss")
  desc:add_input(logits)
  desc:add_input(labels)
  desc:set_type("T", dtype)
  desc:set_type("Tlabels", labels_dtype)
  local loss = desc:finish()
  if options.per_example then return loss end
  if options.class_weights ~= nil then
    local weights = options.class_weights
    if type(weights) ~= "table" then
      error("softmax_cross_entropy class_weights must be a table", 3)
    end
    local classes = 0
    for key in pairs(weights) do
      if type(key) ~= "number" then
        error("softmax_cross_entropy class_weights keys must be numeric", 3)
      end
      classes = math.max(classes, key + 1)
    end
    local values = {}
    for index = 0, classes - 1 do values[#values + 1] = weights[index] or 1 end
    local vector = tf().constant(graph, values, prefix .. "/class_weights",
      dtype)
    local per_class = tf().mul(graph, labels, vector,
      prefix .. "/class_weighted_labels", dtype)
    local per_example = tf().reduce_sum(graph, per_class, { 1 },
      prefix .. "/class_weight", dtype, false)
    local weighted = tf().mul(graph, loss, per_example,
      prefix .. "/weighted", dtype)
    local total = tf().reduce_sum(graph, per_example, { 0 },
      prefix .. "/weight_total", dtype, false)
    local epsilon = tf().constant(graph, 1e-8, prefix .. "/epsilon", dtype)
    return tf().div(graph, tf().reduce_sum(graph, weighted, options.axes or { 0 },
      prefix .. "/weighted_sum", dtype, options.keep_dims), tf().maximum(graph,
      total, epsilon, prefix .. "/safe_total", dtype), prefix .. "/mean", dtype)
  end
  return tf().reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.softmax_cross_entropy_with_logits(graph, logits, labels,
    options)
  return Losses.softmax_cross_entropy(graph, logits, labels, options)
end

return Losses
