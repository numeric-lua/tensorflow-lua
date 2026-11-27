--
-- LuaJIT TensorFlow C API runtime.
--
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")
local Graph = require("tensorflow.graph")
local Session = require("tensorflow.session")
local GraphBuilder = require("tensorflow.graph_builder")
local Train = require("tensorflow.train")
local Checkpoint = require("tensorflow.checkpoint")
local CheckpointReader = require("tensorflow.checkpoint_reader")
local Ops = require("tensorflow.ops")
local Eager = require("tensorflow.eager")
local Layers = require("tensorflow.layers")
local Losses = require("tensorflow.losses")
local Metrics = require("tensorflow.metrics")
local Schedules = require("tensorflow.schedules")
local Functional = require("tensorflow.functional")
local Bundle = require("tensorflow.bundle")
local Sequential = require("tensorflow.model")
local Saver = require("tensorflow.saver")
local Data = require("tensorflow.data")
local Callbacks = require("tensorflow.callbacks")

local tf = {
  VERSION = "2.8.0",
  Tensor = Tensor,
  Graph = Graph,
  Session = Session,
  tensor = Tensor.new,
  complex = Tensor.complex,
}

function tf.placeholder(graph, dtype, shape, name)
  return GraphBuilder.placeholder(graph, dtype, shape, name)
end

function tf.constant(graph, value, name, dtype)
  return Train.constant(graph, value, name, dtype)
end

function tf.variable(graph, initial, name, options)
  return Train.variable(graph, initial, name, options)
end

tf.CheckpointReader = CheckpointReader

function tf.save_checkpoint(session, path, variables)
  return Checkpoint.save(session, path, variables)
end

function tf.save_v2(graph, prefix, variables, name)
  return Train.save_v2(graph, prefix, variables, name)
end

function tf.restore_v2(graph, prefix, variables, name)
  return Train.restore_v2(graph, prefix, variables, name)
end

function tf.load_checkpoint(graph, path, variables)
  return Checkpoint.load(graph, path, variables)
end

function tf.checkpoint_metadata(path)
  return Checkpoint.metadata(path)
end

tf.train = {
  Saver = Saver,
  gradient_descent = function(graph, loss, variables, lr, name, options)
    return Train.gradient_descent(graph, loss, variables, lr, name, options)
  end,
  sgd = function(graph, loss, variables, options)
    return Train.sgd(graph, loss, variables, options)
  end,
  optimizer = function(kind, graph, loss, variables, options)
    return Train.optimizer(kind, graph, loss, variables, options)
  end,
  exponential_moving_average = function(graph, variables, decay, options)
    return Train.exponential_moving_average(graph, variables, decay, options)
  end,
  adam = function(graph, loss, variables, options)
    return Train.adam(graph, loss, variables, options)
  end,
  adamw = function(graph, loss, variables, options)
    return Train.adamw(graph, loss, variables, options)
  end,
  adamax = function(graph, loss, variables, options)
    return Train.adamax(graph, loss, variables, options)
  end,
  rmsprop = function(graph, loss, variables, options)
    return Train.rmsprop(graph, loss, variables, options)
  end,
  centered_rmsprop = function(graph, loss, variables, options)
    return Train.centered_rmsprop(graph, loss, variables, options)
  end,
  adagrad = function(graph, loss, variables, options)
    return Train.adagrad(graph, loss, variables, options)
  end,
  adadelta = function(graph, loss, variables, options)
    return Train.adadelta(graph, loss, variables, options)
  end,
  ftrl = function(graph, loss, variables, options)
    return Train.ftrl(graph, loss, variables, options)
  end,
  proximal_adagrad = function(graph, loss, variables, options)
    return Train.proximal_adagrad(graph, loss, variables, options)
  end,
  momentum = function(graph, loss, variables, options)
    return Train.momentum(graph, loss, variables, options)
  end,
  clip_by_norm = function(graph, gradient, clip_norm, name, dtype)
    return Train.clip_by_norm(graph, gradient, clip_norm, name, dtype)
  end,
  clip_by_global_norm = function(graph, gradients, clip_norm, name, dtype)
    return Train.clip_by_global_norm(graph, gradients, clip_norm, name, dtype)
  end,
}

function tf.gradients(graph, outputs, inputs, grad_ys)
  return GraphBuilder.gradients(graph, outputs, inputs, grad_ys)
end

function tf.add(graph, a, b, name, dtype)
  return GraphBuilder.add(graph, a, b, name, dtype)
end

function tf.matmul(graph, a, b, name, dtype)
  return GraphBuilder.matmul(graph, a, b, name, dtype)
end

function tf.sub(graph, a, b, name, dtype)
  return GraphBuilder.sub(graph, a, b, name, dtype)
end

function tf.mul(graph, a, b, name, dtype)
  return GraphBuilder.mul(graph, a, b, name, dtype)
end

function tf.square(graph, a, name, dtype)
  return GraphBuilder.square(graph, a, name, dtype)
end

tf.nn = {
  relu = Ops.relu,
  relu6 = Ops.relu6,
  leaky_relu = Ops.leaky_relu,
  elu = Ops.elu,
  softsign = Ops.softsign,
  selu = Ops.selu,
  sigmoid = Ops.sigmoid,
  tanh = Ops.tanh,
  softmax = Ops.softmax,
  log_softmax = Ops.log_softmax,
  softplus = Ops.softplus,
}

tf.eager = Eager
tf.layers = Layers
tf.losses = Losses
tf.metrics = Metrics
tf.schedules = Schedules
tf.Functional = Functional
tf.save_bundle = Bundle.save
tf.load_bundle = Bundle.load
tf.Sequential = Sequential
tf.callbacks = Callbacks
tf.data = Data
tf.depth_to_space = Ops.depth_to_space
tf.space_to_depth = Ops.space_to_depth
tf.space_to_batch_nd = Ops.space_to_batch_nd
tf.batch_to_space_nd = Ops.batch_to_space_nd
tf.reshape = Ops.reshape
tf.transpose = Ops.transpose
tf.expand_dims = Ops.expand_dims
tf.squeeze = Ops.squeeze
tf.split = Ops.split
tf.pad = Ops.pad
tf.tile = Ops.tile
tf.broadcast_to = Ops.broadcast_to
tf.roll = Ops.roll
tf.where = Ops.where
tf.stack = Ops.stack
tf.unstack = Ops.unstack
tf.slice = Ops.slice
tf.reverse = Ops.reverse
tf.invert_permutation = Ops.invert_permutation
tf.reverse_sequence = Ops.reverse_sequence
tf.gather = Ops.gather
tf.boolean_mask = Ops.boolean_mask
tf.gather_nd = Ops.gather_nd
tf.scatter_nd = Ops.scatter_nd
tf.tensor_scatter_nd_update = Ops.tensor_scatter_nd_update
tf.tensor_scatter_nd_add = Ops.tensor_scatter_nd_add
tf.tensor_scatter_nd_sub = Ops.tensor_scatter_nd_sub
tf.tensor_scatter_nd_min = Ops.tensor_scatter_nd_min
tf.tensor_scatter_nd_max = Ops.tensor_scatter_nd_max
tf.unique = Ops.unique
tf.unique_with_counts = Ops.unique_with_counts
tf.sequence_mask = Ops.sequence_mask
tf.cast = Ops.cast
tf.clip_by_value = Ops.clip_by_value
tf.maximum = Ops.maximum
tf.minimum = Ops.minimum
tf.squared_difference = Ops.squared_difference
tf.equal = Ops.equal
tf.not_equal = Ops.not_equal
tf.greater = Ops.greater
tf.greater_equal = Ops.greater_equal
tf.less = Ops.less
tf.less_equal = Ops.less_equal
tf.logical_and = Ops.logical_and
tf.logical_or = Ops.logical_or
tf.logical_xor = Ops.logical_xor
tf.logical_not = Ops.logical_not
tf.shape = Ops.shape
tf.div = Ops.div
tf.concat = Ops.concat
tf.abs = Ops.abs
tf.add_n = Ops.add_n
tf.zeros_like = Ops.zeros_like
tf.ones_like = Ops.ones_like
tf.diag = Ops.diag
tf.diag_part = Ops.diag_part
tf.matrix_diag = Ops.matrix_diag
tf.matrix_diag_part = Ops.matrix_diag_part
tf.matrix_band_part = Ops.matrix_band_part
tf.matrix_set_diag = Ops.matrix_set_diag
tf.matrix_inverse = Ops.matrix_inverse
tf.matrix_determinant = Ops.matrix_determinant
tf.matrix_solve = Ops.matrix_solve
tf.cholesky = Ops.cholesky
tf.matrix_triangular_solve = Ops.matrix_triangular_solve
tf.qr = Ops.qr
tf.self_adjoint_eig = Ops.self_adjoint_eig
tf.svd = Ops.svd
tf.identity = Ops.identity
tf.stop_gradient = Ops.stop_gradient
tf.batch_matmul = Ops.batch_matmul
tf.fill = Ops.fill
tf.zeros = Ops.zeros
tf.ones = Ops.ones
tf.range = Ops.range
tf.linspace = Ops.linspace
tf.reduce_mean = Ops.reduce_mean
tf.reduce_sum = Ops.reduce_sum
tf.reduce_logsumexp = Ops.reduce_logsumexp
tf.reduce_prod = Ops.reduce_prod
tf.reduce_max = Ops.reduce_max
tf.reduce_min = Ops.reduce_min
tf.reduce_variance = Ops.reduce_variance
tf.reduce_stddev = Ops.reduce_stddev
tf.norm = Ops.norm
tf.reduce_all = Ops.reduce_all
tf.reduce_any = Ops.reduce_any
tf.argmax = Ops.argmax
tf.argmin = Ops.argmin
tf.top_k = Ops.top_k
tf.in_top_k = Ops.in_top_k
tf.bincount = Ops.bincount
tf.one_hot = Ops.one_hot
tf.exp = Ops.exp
tf.log = Ops.log
tf.sqrt = Ops.sqrt
tf.erf = Ops.erf
tf.erfc = Ops.erfc
tf.is_nan = Ops.is_nan
tf.is_inf = Ops.is_inf
tf.is_finite = Ops.is_finite
tf.floor = Ops.floor
tf.ceil = Ops.ceil
tf.round = Ops.round
tf.sign = Ops.sign
tf.rsqrt = Ops.rsqrt
tf.reciprocal = Ops.reciprocal

function tf.version()
  capi.load()
  return capi.version
end

function tf.load_saved_model(path, tags, options)
  return Session.load_saved_model(path, tags, options)
end

return tf
