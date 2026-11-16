--
-- LuaJIT TensorFlow C API runtime.
--
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")
local Graph = require("tensorflow.graph")
local Session = require("tensorflow.session")
local GraphBuilder = require("tensorflow.graph_builder")

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

function tf.add(graph, a, b, name, dtype)
  return GraphBuilder.add(graph, a, b, name, dtype)
end

function tf.matmul(graph, a, b, name, dtype)
  return GraphBuilder.matmul(graph, a, b, name, dtype)
end

function tf.version()
  capi.load()
  return capi.version
end

function tf.load_saved_model(path, tags, options)
  return Session.load_saved_model(path, tags, options)
end

return tf
