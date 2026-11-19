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
local Sequential = require("tensorflow.model")
local Saver = require("tensorflow.saver")
local Data = require("tensorflow.data")
local Callbacks = require("tensorflow.callbacks")

local tf = {
  VERSION = "0.15.0",
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

tf.train = {
  Saver = Saver,
  gradient_descent = function(graph, loss, variables, lr, name)
    return Train.gradient_descent(graph, loss, variables, lr, name)
  end,
  adam = function(graph, loss, variables, options)
    return Train.adam(graph, loss, variables, options)
  end,
  adamw = function(graph, loss, variables, options)
    return Train.adamw(graph, loss, variables, options)
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
}

function tf.gradients(graph, outputs, inputs)
  return GraphBuilder.gradients(graph, outputs, inputs)
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
  leaky_relu = Ops.leaky_relu,
  elu = Ops.elu,
  sigmoid = Ops.sigmoid,
  tanh = Ops.tanh,
  softmax = Ops.softmax,
  softplus = Ops.softplus,
}

tf.eager = Eager
tf.layers = Layers
tf.losses = Losses
tf.metrics = Metrics
tf.schedules = Schedules
tf.Functional = Functional
tf.Sequential = Sequential
tf.callbacks = Callbacks
tf.data = Data
tf.reshape = Ops.reshape
tf.cast = Ops.cast
tf.clip_by_value = Ops.clip_by_value
tf.maximum = Ops.maximum
tf.minimum = Ops.minimum
tf.shape = Ops.shape
tf.div = Ops.div
tf.concat = Ops.concat
tf.abs = Ops.abs
tf.reduce_mean = Ops.reduce_mean
tf.reduce_sum = Ops.reduce_sum
tf.reduce_max = Ops.reduce_max
tf.reduce_min = Ops.reduce_min
tf.argmax = Ops.argmax
tf.one_hot = Ops.one_hot
tf.exp = Ops.exp
tf.log = Ops.log
tf.sqrt = Ops.sqrt

function tf.version()
  capi.load()
  return capi.version
end

function tf.load_saved_model(path, tags, options)
  return Session.load_saved_model(path, tags, options)
end

return tf
