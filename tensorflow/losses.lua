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
  return tf().reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
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
  return tf().reduce_mean(graph, loss, options.axes or { 0 },
    prefix .. "/mean", dtype, options.keep_dims)
end

function Losses.softmax_cross_entropy_with_logits(graph, logits, labels,
    options)
  return Losses.softmax_cross_entropy(graph, logits, labels, options)
end

return Losses
