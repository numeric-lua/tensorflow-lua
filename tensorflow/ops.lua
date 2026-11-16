--
-- Typed convenience ops built on the generic graph builder.
--
local Tensor = require("tensorflow.tensor")

local Ops = {}

local function dtype_name(value)
  if type(value) == "table" and value.dtype then return value.dtype end
  return value or "float32"
end

local function constant(graph, value, name, dtype)
  dtype = dtype_name(dtype)
  local tensor = getmetatable(value) == Tensor and value or
    Tensor.new(value, { dtype = dtype })
  local desc = graph:new_operation("Const", name)
  desc:set_tensor("value", tensor)
  desc:set_type("dtype", dtype)
  return desc:finish()
end

local function unary(graph, op_type, input, name, dtype)
  local desc = graph:new_operation(op_type, name)
  desc:add_input(input)
  desc:set_type("T", dtype_name(dtype))
  return desc:finish()
end

function Ops.relu(graph, input, name, dtype)
  return unary(graph, "Relu", input, name, dtype)
end

function Ops.sigmoid(graph, input, name, dtype)
  return unary(graph, "Sigmoid", input, name, dtype)
end

function Ops.tanh(graph, input, name, dtype)
  return unary(graph, "Tanh", input, name, dtype)
end

function Ops.exp(graph, input, name, dtype)
  return unary(graph, "Exp", input, name, dtype)
end

function Ops.log(graph, input, name, dtype)
  return unary(graph, "Log", input, name, dtype)
end

function Ops.sqrt(graph, input, name, dtype)
  return unary(graph, "Sqrt", input, name, dtype)
end

function Ops.softmax(graph, input, name, dtype)
  return unary(graph, "Softmax", input, name, dtype)
end

function Ops.reshape(graph, input, shape, name, dtype)
  if type(shape) ~= "table" then
    error("reshape expects a shape table", 3)
  end
  local shape_op = constant(graph, shape, name .. "/shape", "int32")
  local desc = graph:new_operation("Reshape", name)
  desc:add_input(input)
  desc:add_input(shape_op)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tshape", "int32")
  return desc:finish()
end

function Ops.reduce_mean(graph, input, axes, name, dtype, keep_dims)
  local axis_op = constant(graph, axes or {}, name .. "/axes", "int32")
  local desc = graph:new_operation("Mean", name)
  desc:add_input(input)
  desc:add_input(axis_op)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tidx", "int32")
  desc:set_bool("keep_dims", keep_dims and true or false)
  return desc:finish()
end

function Ops.argmax(graph, input, axis, name, dtype, output_dtype)
  local dimension = constant(graph, axis or 0, name .. "/dimension", "int32")
  local desc = graph:new_operation("ArgMax", name)
  desc:add_input(input)
  desc:add_input(dimension)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tidx", "int32")
  desc:set_type("output_type", output_dtype or "int64")
  return desc:finish()
end

return Ops
