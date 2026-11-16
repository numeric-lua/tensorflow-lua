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

function Ops.elu(graph, input, name, dtype)
  return unary(graph, "Elu", input, name, dtype)
end

function Ops.leaky_relu(graph, input, name, dtype, alpha)
  local desc = graph:new_operation("LeakyRelu", name)
  desc:add_input(input)
  desc:set_type("T", dtype_name(dtype))
  desc:set_float("alpha", alpha or 0.2)
  return desc:finish()
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

function Ops.abs(graph, input, name, dtype)
  return unary(graph, "Abs", input, name, dtype)
end

function Ops.softplus(graph, input, name, dtype)
  return unary(graph, "Softplus", input, name, dtype)
end

function Ops.sqrt(graph, input, name, dtype)
  return unary(graph, "Sqrt", input, name, dtype)
end

function Ops.softmax(graph, input, name, dtype)
  return unary(graph, "Softmax", input, name, dtype)
end

function Ops.shape(graph, input, name, dtype)
  local desc = graph:new_operation("Shape", name)
  desc:add_input(input)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("out_type", "int32")
  return desc:finish()
end

function Ops.div(graph, a, b, name, dtype)
  local desc = graph:new_operation("RealDiv", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype_name(dtype))
  return desc:finish()
end

function Ops.clip_by_value(graph, input, minimum, maximum, name, dtype)
  local prefix = name or "clip_by_value"
  dtype = dtype_name(dtype)
  local desc = graph:new_operation("ClipByValue", prefix)
  desc:add_input(input)
  desc:add_input(constant(graph, minimum, prefix .. "/minimum", dtype))
  desc:add_input(constant(graph, maximum, prefix .. "/maximum", dtype))
  desc:set_type("T", dtype)
  return desc:finish()
end

function Ops.cast(graph, input, src_dtype, dst_dtype, name)
  local desc = graph:new_operation("Cast", name)
  desc:add_input(input)
  desc:set_type("SrcT", dtype_name(src_dtype))
  desc:set_type("DstT", dtype_name(dst_dtype))
  return desc:finish()
end

function Ops.concat(graph, inputs, axis, name, dtype)
  if type(inputs) ~= "table" or #inputs < 2 then
    error("concat expects at least two inputs", 3)
  end
  local desc = graph:new_operation("ConcatV2", name)
  desc:add_inputs(inputs)
  desc:add_input(constant(graph, axis or 0, name .. "/axis", "int32"))
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tidx", "int32")
  desc:set_int("N", #inputs)
  return desc:finish()
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

local function reduction(graph, op_type, input, axes, name, dtype, keep_dims)
  local axis_op = constant(graph, axes or {}, name .. "/axes", "int32")
  local desc = graph:new_operation(op_type, name)
  desc:add_input(input)
  desc:add_input(axis_op)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tidx", "int32")
  desc:set_bool("keep_dims", keep_dims and true or false)
  return desc:finish()
end

function Ops.reduce_mean(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Mean", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_sum(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Sum", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_max(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Max", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_min(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Min", input, axes, name, dtype, keep_dims)
end

function Ops.one_hot(graph, indices, depth, options)
  options = options or {}
  local dtype = dtype_name(options.dtype or "float32")
  local prefix = options.name or "one_hot"
  local depth_op = constant(graph, depth, prefix .. "/depth", "int32")
  local on = constant(graph, options.on_value or 1, prefix .. "/on", dtype)
  local off = constant(graph, options.off_value or 0, prefix .. "/off", dtype)
  local desc = graph:new_operation("OneHot", prefix)
  desc:add_input(indices)
  desc:add_input(depth_op)
  desc:add_input(on)
  desc:add_input(off)
  desc:set_type("T", dtype)
  desc:set_type("TI", options.indices_dtype or "int32")
  desc:set_int("axis", options.axis or -1)
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
