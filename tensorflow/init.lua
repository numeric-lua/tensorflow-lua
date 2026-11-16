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

local tf = {
  VERSION = "0.1.0",
  Tensor = Tensor,
  Graph = Graph,
  Session = Session,
  tensor = Tensor.new,
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

function tf.load_checkpoint(graph, path, variables)
  return Checkpoint.load(graph, path, variables)
end

tf.train = {
  gradient_descent = function(graph, loss, variables, lr, name)
    return Train.gradient_descent(graph, loss, variables, lr, name)
  end,
  adam = function(graph, loss, variables, options)
    return Train.adam(graph, loss, variables, options)
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

function tf.version()
  capi.load()
  return capi.version
end

function tf.load_saved_model(path, tags, options)
  return Session.load_saved_model(path, tags, options)
end

return tf
