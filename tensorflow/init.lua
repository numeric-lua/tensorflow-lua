--
-- LuaJIT TensorFlow C API runtime.
--
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")
local Graph = require("tensorflow.graph")
local Session = require("tensorflow.session")

local tf = {
  VERSION = "0.1.0",
  Tensor = Tensor,
  Graph = Graph,
  Session = Session,
  tensor = Tensor.new,
}

function tf.version()
  capi.load()
  return capi.version
end

function tf.load_saved_model(path, tags, options)
  return Session.load_saved_model(path, tags, options)
end

return tf
