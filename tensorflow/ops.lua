--
-- Typed convenience ops built on the generic graph builder.
--
local Tensor = require("tensorflow.tensor")

local Ops = {}

local function dtype_name(value)
  if type(value) == "table" and value.dtype then return value.dtype end
  return value or "float32"
end

local function input_dtype(input, dtype)
  if dtype ~= nil then return dtype_name(dtype) end
  if type(input) == "table" and input.dtype ~= nil then
    return dtype_name(input.dtype)
  end
  return "float32"
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

local function typed_unary(graph, op_type, input, name, dtype)
  local value_dtype = input_dtype(input, dtype)
  local output = unary(graph, op_type, input, name, value_dtype)
  output.dtype = value_dtype
  return output
end

function Ops.relu(graph, input, name, dtype)
  return unary(graph, "Relu", input, name, dtype)
end

function Ops.relu6(graph, input, name, dtype)
  local output = unary(graph, "Relu6", input, name, dtype)
  output.dtype = dtype_name(dtype)
  return output
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

function Ops.floor(graph, input, name, dtype)
  return typed_unary(graph, "Floor", input, name, dtype)
end

function Ops.ceil(graph, input, name, dtype)
  return typed_unary(graph, "Ceil", input, name, dtype)
end

function Ops.round(graph, input, name, dtype)
  return typed_unary(graph, "Round", input, name, dtype)
end

function Ops.sign(graph, input, name, dtype)
  return typed_unary(graph, "Sign", input, name, dtype)
end

function Ops.rsqrt(graph, input, name, dtype)
  return typed_unary(graph, "Rsqrt", input, name, dtype)
end

function Ops.reciprocal(graph, input, name, dtype)
  return typed_unary(graph, "Reciprocal", input, name, dtype)
end

function Ops.softmax(graph, input, name, dtype)
  return unary(graph, "Softmax", input, name, dtype)
end

function Ops.log_softmax(graph, input, name, dtype)
  local value_dtype = input_dtype(input, dtype)
  local output = unary(graph, "LogSoftmax", input, name, value_dtype)
  output.dtype = value_dtype
  return output
end

function Ops.fill(graph, dims, value, name, dtype)
  if type(dims) ~= "table" then
    error("fill expects a dims table", 3)
  end
  for i, dimension in ipairs(dims) do
    if type(dimension) ~= "number" or dimension % 1 ~= 0 or dimension < 0 then
      error("fill dimensions must be non-negative integers (dimension " ..
        tostring(i) .. ")", 3)
    end
  end
  name = name or "fill"
  local value_dtype = dtype_name(dtype)
  local dims_op = constant(graph, dims, name .. "/dims", "int32")
  local value_op = constant(graph, value, name .. "/value", value_dtype)
  local desc = graph:new_operation("Fill", name)
  desc:add_input(dims_op)
  desc:add_input(value_op)
  desc:set_type("T", value_dtype)
  desc:set_type("index_type", "int32")
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.range(graph, start, limit, delta, name, dtype)
  name = name or "range"
  local value_dtype = dtype_name(dtype)
  local start_op = constant(graph, start, name .. "/start", value_dtype)
  local limit_op = constant(graph, limit, name .. "/limit", value_dtype)
  local delta_op = constant(graph, delta, name .. "/delta", value_dtype)
  local desc = graph:new_operation("Range", name)
  desc:add_input(start_op)
  desc:add_input(limit_op)
  desc:add_input(delta_op)
  desc:set_type("Tidx", value_dtype)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.linspace(graph, start, stop, num, name, dtype)
  local value_dtype = dtype_name(dtype)
  if value_dtype ~= "float32" and value_dtype ~= "float64" then
    error("linspace dtype must be float32 or float64", 3)
  end
  if type(num) ~= "number" or num % 1 ~= 0 or num < 1 then
    error("linspace num must be a positive integer", 3)
  end
  name = name or "linspace"
  local start_op = constant(graph, start, name .. "/start", value_dtype)
  local stop_op = constant(graph, stop, name .. "/stop", value_dtype)
  local num_op = constant(graph, num, name .. "/num", "int32")
  local desc = graph:new_operation("LinSpace", name)
  desc:add_input(start_op)
  desc:add_input(stop_op)
  desc:add_input(num_op)
  desc:set_type("T", value_dtype)
  desc:set_type("Tidx", "int32")
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.batch_matmul(graph, a, b, name, dtype, options)
  options = options or {}
  name = name or "batch_matmul"
  local adj_x = options.adj_x
  if adj_x == nil then adj_x = options.transpose_a end
  local adj_y = options.adj_y
  if adj_y == nil then adj_y = options.transpose_b end
  if adj_x ~= nil and type(adj_x) ~= "boolean" then
    error("batch_matmul adj_x must be a boolean", 3)
  end
  if adj_y ~= nil and type(adj_y) ~= "boolean" then
    error("batch_matmul adj_y must be a boolean", 3)
  end
  local value_dtype = input_dtype(a, dtype)
  local desc = graph:new_operation("BatchMatMulV2", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", value_dtype)
  desc:set_bool("adj_x", adj_x == true)
  desc:set_bool("adj_y", adj_y == true)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
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

function Ops.maximum(graph, a, b, name, dtype)
  local desc = graph:new_operation("Maximum", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.minimum(graph, a, b, name, dtype)
  local desc = graph:new_operation("Minimum", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
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

function Ops.split(graph, input, num_or_size_splits, axis, name, dtype)
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("split expects an integer axis", 3)
  end
  name = name or "split"
  local value_dtype = input_dtype(input, dtype)
  local desc
  local count
  if type(num_or_size_splits) == "number" then
    if num_or_size_splits % 1 ~= 0 or num_or_size_splits < 1 then
      error("split expects a positive integer split count", 3)
    end
    count = num_or_size_splits
    local axis_op = constant(graph, axis, name .. "/axis", "int32")
    desc = graph:new_operation("Split", name)
    desc:add_input(axis_op)
    desc:add_input(input)
    desc:set_type("T", value_dtype)
    desc:set_int("num_split", count)
  elseif type(num_or_size_splits) == "table" then
    if #num_or_size_splits == 0 then
      error("split expects a non-empty size table", 3)
    end
    for _, size in ipairs(num_or_size_splits) do
      if type(size) ~= "number" or size % 1 ~= 0 or size < 0 then
        error("split sizes must be non-negative integers", 3)
      end
    end
    count = #num_or_size_splits
    local sizes = constant(graph, num_or_size_splits,
      name .. "/sizes", "int64")
    local axis_op = constant(graph, axis, name .. "/axis", "int32")
    desc = graph:new_operation("SplitV", name)
    desc:add_input(input)
    desc:add_input(sizes)
    desc:add_input(axis_op)
    desc:set_type("T", value_dtype)
    desc:set_type("Tlen", "int64")
    desc:set_int("num_split", count)
  else
    error("split expects a split count or size table", 3)
  end
  local result = desc:finish()
  local outputs = {}
  for i = 1, count do
    outputs[i] = {
      oper = result.oper,
      index = i - 1,
      name = name .. ":" .. (i - 1),
      dtype = value_dtype,
    }
  end
  return outputs
end

function Ops.pad(graph, input, paddings, mode, name, dtype)
  if type(paddings) ~= "table" or #paddings == 0 then
    error("pad expects a non-empty paddings table", 3)
  end
  for _, row in ipairs(paddings) do
    if type(row) ~= "table" or #row ~= 2 then
      error("pad paddings must be pairs of integers", 3)
    end
    for _, amount in ipairs(row) do
      if type(amount) ~= "number" or amount % 1 ~= 0 or amount < 0 then
        error("pad amounts must be non-negative integers", 3)
      end
    end
  end
  mode = string.upper(mode or "CONSTANT")
  if mode ~= "CONSTANT" and mode ~= "REFLECT" and mode ~= "SYMMETRIC" then
    error("pad mode must be CONSTANT, REFLECT, or SYMMETRIC", 3)
  end
  name = name or "pad"
  local value_dtype = input_dtype(input, dtype)
  local padding_op = constant(graph, paddings, name .. "/paddings", "int32")
  local desc
  if mode == "CONSTANT" then
    desc = graph:new_operation("Pad", name)
  else
    desc = graph:new_operation("MirrorPad", name)
    desc:set_string("mode", mode)
  end
  desc:add_input(input)
  desc:add_input(padding_op)
  desc:set_type("T", value_dtype)
  desc:set_type("Tpaddings", "int32")
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.tile(graph, input, multiples, name, dtype)
  if type(multiples) ~= "table" or #multiples == 0 then
    error("tile expects a non-empty multiples table", 3)
  end
  for _, multiple in ipairs(multiples) do
    if type(multiple) ~= "number" or multiple % 1 ~= 0 or multiple < 0 then
      error("tile multiples must be non-negative integers", 3)
    end
  end
  name = name or "tile"
  local value_dtype = input_dtype(input, dtype)
  local multiples_op = constant(graph, multiples, name .. "/multiples",
    "int32")
  local desc = graph:new_operation("Tile", name)
  desc:add_input(input)
  desc:add_input(multiples_op)
  desc:set_type("T", value_dtype)
  desc:set_type("Tmultiples", "int32")
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.where(graph, condition, x, y, name, dtype)
  name = name or "where"
  local value_dtype = input_dtype(x, dtype)
  local desc = graph:new_operation("SelectV2", name)
  desc:add_input(condition)
  desc:add_input(x)
  desc:add_input(y)
  desc:set_type("T", value_dtype)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.stack(graph, inputs, axis, name, dtype)
  if type(inputs) ~= "table" or #inputs == 0 then
    error("stack expects a non-empty input list", 3)
  end
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("stack expects an integer axis", 3)
  end
  name = name or "stack"
  local value_dtype = input_dtype(inputs[1], dtype)
  local desc = graph:new_operation("Pack", name)
  desc:add_inputs(inputs)
  desc:set_type("T", value_dtype)
  desc:set_int("N", #inputs)
  desc:set_int("axis", axis)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.unstack(graph, input, num, axis, name, dtype)
  if type(num) ~= "number" or num % 1 ~= 0 or num < 1 then
    error("unstack expects a positive integer count", 3)
  end
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("unstack expects an integer axis", 3)
  end
  name = name or "unstack"
  local value_dtype = input_dtype(input, dtype)
  local desc = graph:new_operation("Unpack", name)
  desc:add_input(input)
  desc:set_type("T", value_dtype)
  desc:set_int("num", num)
  desc:set_int("axis", axis)
  local result = desc:finish()
  local outputs = {}
  for i = 1, num do
    outputs[i] = {
      oper = result.oper,
      index = i - 1,
      name = name .. ":" .. (i - 1),
      dtype = value_dtype,
    }
  end
  return outputs
end

function Ops.slice(graph, input, begin, size, name, dtype)
  if type(begin) ~= "table" or #begin == 0 or type(size) ~= "table" or
      #size ~= #begin then
    error("slice expects matching non-empty begin and size tables", 3)
  end
  for i = 1, #begin do
    if type(begin[i]) ~= "number" or begin[i] % 1 ~= 0 then
      error("slice begin values must be integers", 3)
    end
    if type(size[i]) ~= "number" or size[i] % 1 ~= 0 or size[i] < -1 then
      error("slice size values must be integers greater than or equal to -1", 3)
    end
  end
  name = name or "slice"
  local value_dtype = input_dtype(input, dtype)
  local begin_op = constant(graph, begin, name .. "/begin", "int32")
  local size_op = constant(graph, size, name .. "/size", "int32")
  local desc = graph:new_operation("Slice", name)
  desc:add_input(input)
  desc:add_input(begin_op)
  desc:add_input(size_op)
  desc:set_type("T", value_dtype)
  desc:set_type("Index", "int32")
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.reverse(graph, input, axis, name, dtype)
  if type(axis) == "number" then axis = { axis } end
  if type(axis) ~= "table" or #axis == 0 then
    error("reverse expects an integer axis or non-empty axis table", 3)
  end
  for _, dimension in ipairs(axis) do
    if type(dimension) ~= "number" or dimension % 1 ~= 0 then
      error("reverse axes must be integers", 3)
    end
  end
  name = name or "reverse"
  local value_dtype = input_dtype(input, dtype)
  local axis_op = constant(graph, axis, name .. "/axis", "int32")
  local desc = graph:new_operation("ReverseV2", name)
  desc:add_input(input)
  desc:add_input(axis_op)
  desc:set_type("T", value_dtype)
  desc:set_type("Tidx", "int32")
  local output = desc:finish()
  output.dtype = value_dtype
  return output
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

function Ops.transpose(graph, input, perm, name, dtype)
  if type(perm) ~= "table" or #perm == 0 then
    error("transpose expects a non-empty permutation table", 3)
  end
  name = name or "transpose"
  local perm_op = constant(graph, perm, name .. "/perm", "int32")
  local desc = graph:new_operation("Transpose", name)
  desc:add_input(input)
  desc:add_input(perm_op)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tperm", "int32")
  return desc:finish()
end

function Ops.expand_dims(graph, input, axis, name, dtype)
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("expand_dims expects an integer axis", 3)
  end
  name = name or "expand_dims"
  local axis_op = constant(graph, axis, name .. "/axis", "int32")
  local desc = graph:new_operation("ExpandDims", name)
  desc:add_input(input)
  desc:add_input(axis_op)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tdim", "int32")
  return desc:finish()
end

function Ops.squeeze(graph, input, axis, name, dtype)
  if axis ~= nil and type(axis) ~= "number" and type(axis) ~= "table" then
    error("squeeze expects an integer axis or table", 3)
  end
  if type(axis) == "number" then axis = { axis } end
  axis = axis or {}
  for _, dimension in ipairs(axis) do
    if type(dimension) ~= "number" or dimension % 1 ~= 0 then
      error("squeeze axes must be integers", 3)
    end
  end
  name = name or "squeeze"
  local desc = graph:new_operation("Squeeze", name)
  desc:add_input(input)
  desc:set_type("T", dtype_name(dtype))
  if #axis > 0 then desc:set_int_list("squeeze_dims", axis) end
  return desc:finish()
end

function Ops.gather(graph, params, indices, axis, name, dtype, indices_dtype)
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("gather expects an integer axis", 3)
  end
  name = name or "gather"
  local axis_op = constant(graph, axis, name .. "/axis", "int32")
  local index_dtype = dtype_name(indices_dtype or
    (type(indices) == "table" and indices.dtype) or "int32")
  local desc = graph:new_operation("GatherV2", name)
  desc:add_input(params)
  desc:add_input(indices)
  desc:add_input(axis_op)
  desc:set_type("Tparams", dtype_name(dtype))
  desc:set_type("Tindices", index_dtype)
  desc:set_type("Taxis", "int32")
  desc:set_int("batch_dims", 0)
  return desc:finish()
end

local function comparison(graph, op_type, a, b, name, dtype)
  local value_dtype = input_dtype(a, dtype)
  local desc = graph:new_operation(op_type, name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", value_dtype)
  local output = desc:finish()
  output.dtype = "bool"
  return output
end

function Ops.equal(graph, a, b, name, dtype)
  return comparison(graph, "Equal", a, b, name, dtype)
end

function Ops.not_equal(graph, a, b, name, dtype)
  return comparison(graph, "NotEqual", a, b, name, dtype)
end

function Ops.greater(graph, a, b, name, dtype)
  return comparison(graph, "Greater", a, b, name, dtype)
end

function Ops.greater_equal(graph, a, b, name, dtype)
  return comparison(graph, "GreaterEqual", a, b, name, dtype)
end

function Ops.less(graph, a, b, name, dtype)
  return comparison(graph, "Less", a, b, name, dtype)
end

function Ops.less_equal(graph, a, b, name, dtype)
  return comparison(graph, "LessEqual", a, b, name, dtype)
end

local function logical_binary(graph, op_type, a, b, name)
  local desc = graph:new_operation(op_type, name)
  desc:add_input(a)
  desc:add_input(b)
  local output = desc:finish()
  output.dtype = "bool"
  return output
end

function Ops.logical_and(graph, a, b, name)
  return logical_binary(graph, "LogicalAnd", a, b, name)
end

function Ops.logical_or(graph, a, b, name)
  return logical_binary(graph, "LogicalOr", a, b, name)
end

function Ops.logical_xor(graph, a, b, name)
  return comparison(graph, "NotEqual", a, b, name)
end

function Ops.logical_not(graph, input, name)
  local desc = graph:new_operation("LogicalNot", name)
  desc:add_input(input)
  local output = desc:finish()
  output.dtype = "bool"
  return output
end

function Ops.squared_difference(graph, a, b, name, dtype)
  local value_dtype = input_dtype(a, dtype)
  local desc = graph:new_operation("SquaredDifference", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", value_dtype)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
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

local function bool_reduction(graph, op_type, input, axes, name, dtype,
    keep_dims)
  local value_dtype = dtype and dtype_name(dtype) or
    (type(input) == "table" and input.dtype) or "bool"
  local axis_op = constant(graph, axes or {}, name .. "/axes", "int32")
  local desc = graph:new_operation(op_type, name)
  desc:add_input(input)
  desc:add_input(axis_op)
  desc:set_type("T", value_dtype)
  desc:set_type("Tidx", "int32")
  desc:set_bool("keep_dims", keep_dims and true or false)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

local function add_output(graph, op_type, a, b, name, dtype)
  local desc = graph:new_operation(op_type, name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  local output = desc:finish()
  output.dtype = dtype
  return output
end

function Ops.add_n(graph, inputs, name, dtype)
  if type(inputs) ~= "table" or #inputs == 0 then
    error("add_n expects a non-empty input list", 3)
  end
  name = name or "add_n"
  local value_dtype = input_dtype(inputs[1], dtype)
  local desc = graph:new_operation("AddN", name)
  desc:add_inputs(inputs)
  desc:set_type("T", value_dtype)
  desc:set_int("N", #inputs)
  local output = desc:finish()
  output.dtype = value_dtype
  return output
end

function Ops.reduce_logsumexp(graph, input, axes, name, dtype, keep_dims)
  axes = axes or {}
  if type(axes) ~= "table" then
    error("reduce_logsumexp expects an axes table", 3)
  end
  for _, axis in ipairs(axes) do
    if type(axis) ~= "number" or axis % 1 ~= 0 then
      error("reduce_logsumexp axes must be integers", 3)
    end
  end
  name = name or "reduce_logsumexp"
  local value_dtype = input_dtype(input, dtype)
  local maximum = Ops.reduce_max(graph, input, axes, name .. "/max",
    value_dtype, true)
  local centered = add_output(graph, "Sub", input, maximum,
    name .. "/centered", value_dtype)
  local exponential = Ops.exp(graph, centered, name .. "/exp", value_dtype)
  local total = Ops.reduce_sum(graph, exponential, axes, name .. "/sum",
    value_dtype, keep_dims)
  local logarithm = Ops.log(graph, total, name .. "/log", value_dtype)
  local reduced_max = keep_dims and maximum or Ops.reduce_max(graph, input,
    axes, name .. "/reduced_max", value_dtype, false)
  return add_output(graph, "AddV2", logarithm, reduced_max, name, value_dtype)
end

function Ops.reduce_mean(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Mean", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_sum(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Sum", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_prod(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Prod", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_max(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Max", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_min(graph, input, axes, name, dtype, keep_dims)
  return reduction(graph, "Min", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_all(graph, input, axes, name, dtype, keep_dims)
  return bool_reduction(graph, "All", input, axes, name, dtype, keep_dims)
end

function Ops.reduce_any(graph, input, axes, name, dtype, keep_dims)
  return bool_reduction(graph, "Any", input, axes, name, dtype, keep_dims)
end

local function variance_axes(axes, label)
  axes = axes or {}
  if type(axes) ~= "table" then
    error(label .. " expects an axes table", 3)
  end
  for _, axis in ipairs(axes) do
    if type(axis) ~= "number" or axis % 1 ~= 0 then
      error(label .. " axes must be integers", 3)
    end
  end
  return axes
end

function Ops.reduce_variance(graph, input, axes, name, dtype, keep_dims)
  axes = variance_axes(axes, "reduce_variance")
  name = name or "reduce_variance"
  local value_dtype = input_dtype(input, dtype)
  local mean = Ops.reduce_mean(graph, input, axes, name .. "/mean",
    value_dtype, true)
  local squared = Ops.squared_difference(graph, input, mean,
    name .. "/squared", value_dtype)
  local variance = Ops.reduce_mean(graph, squared, axes,
    name .. "/variance", value_dtype, keep_dims)
  variance.dtype = value_dtype
  return variance
end

function Ops.reduce_stddev(graph, input, axes, name, dtype, keep_dims)
  axes = variance_axes(axes, "reduce_stddev")
  name = name or "reduce_stddev"
  local value_dtype = input_dtype(input, dtype)
  local variance = Ops.reduce_variance(graph, input, axes, name .. "/variance",
    value_dtype, keep_dims)
  local output = Ops.sqrt(graph, variance, name .. "/sqrt", value_dtype)
  output.dtype = value_dtype
  return output
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

local function arg_reduction(graph, op_type, input, axis, name, dtype,
    output_dtype)
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error(op_type:lower() .. " expects an integer axis", 3)
  end
  local dimension = constant(graph, axis, name .. "/dimension", "int32")
  local desc = graph:new_operation(op_type, name)
  desc:add_input(input)
  desc:add_input(dimension)
  desc:set_type("T", dtype_name(dtype))
  desc:set_type("Tidx", "int32")
  desc:set_type("output_type", output_dtype or "int64")
  local output = desc:finish()
  output.dtype = output_dtype or "int64"
  return output
end

function Ops.argmax(graph, input, axis, name, dtype, output_dtype)
  return arg_reduction(graph, "ArgMax", input, axis, name, dtype, output_dtype)
end

function Ops.argmin(graph, input, axis, name, dtype, output_dtype)
  return arg_reduction(graph, "ArgMin", input, axis, name, dtype, output_dtype)
end

return Ops
