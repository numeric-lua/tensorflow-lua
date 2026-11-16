--
-- TensorFlow graph construction helpers.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")

local GraphBuilder = {}

local function resolve_output(graph, value)
  if type(value) == "table" and value.oper ~= nil then return value end
  if type(value) == "string" then
    local name, index = value, 0
    local base, suffix = value:match("^(.-):(%d+)$")
    if base then name, index = base, tonumber(suffix) end
    return { oper = graph:operation(name), index = index, name = name .. ":" .. index }
  end
  error("operation input must be 'op:index' or a graph output", 3)
end

local Description = {}
Description.__index = Description

local function dtype_code(dtype)
  local _, info = Tensor.dtype_info(dtype)
  return info.code
end

function Description:add_input(value)
  local output = resolve_output(self._graph, value)
  local c = ffi.new("TF_Output")
  c.oper = output.oper
  c.index = output.index
  capi.load().TF_AddInput(self._desc, c)
  return self
end

function Description:add_inputs(values)
  local lib = capi.load()
  local array = ffi.new("TF_Output[?]", #values)
  for i, value in ipairs(values) do
    local output = resolve_output(self._graph, value)
    array[i - 1].oper = output.oper
    array[i - 1].index = output.index
  end
  lib.TF_AddInputList(self._desc, array, #values)
  return self
end

function Description:add_control_input(value)
  local operation
  if type(value) == "table" and value.oper then
    operation = value.oper
  else
    operation = self._graph:operation(value)
  end
  capi.load().TF_AddControlInput(self._desc, operation)
  return self
end

function Description:set_device(device)
  capi.load().TF_SetDevice(self._desc, device)
  return self
end

function Description:set_type(name, dtype)
  capi.load().TF_SetAttrType(self._desc, name, dtype_code(dtype))
  return self
end

function Description:set_int(name, value)
  capi.load().TF_SetAttrInt(self._desc, name, value)
  return self
end

function Description:set_float(name, value)
  capi.load().TF_SetAttrFloat(self._desc, name, value)
  return self
end

function Description:set_bool(name, value)
  capi.load().TF_SetAttrBool(self._desc, name, value and 1 or 0)
  return self
end

function Description:set_string(name, value)
  capi.load().TF_SetAttrString(self._desc, name, value, #value)
  return self
end

function Description:set_shape(name, shape)
  local lib = capi.load()
  if shape == nil then
    lib.TF_SetAttrShape(self._desc, name, nil, -1)
    return self
  end
  local dims = ffi.new("int64_t[?]", #shape)
  for i, value in ipairs(shape) do dims[i - 1] = value end
  lib.TF_SetAttrShape(self._desc, name, dims, #shape)
  return self
end

function Description:set_tensor(name, tensor)
  local status = capi.status()
  capi.load().TF_SetAttrTensor(self._desc, name, tensor:handle(), status)
  capi.check(status, "set tensor attribute")
  self._tensor = tensor
  return self
end

function Description:finish()
  local lib = capi.load()
  local status = capi.status()
  local operand = lib.TF_FinishOperation(self._desc, status)
  local ok, err = pcall(capi.check, status, "finish operation")
  self._desc = nil
  if not ok or operand == nil then error(err or "TF_FinishOperation failed", 2) end
  return { oper = operand, index = 0, name = self._name .. ":0" }
end

function GraphBuilder.new_operation(graph, op_type, name)
  local lib = capi.load()
  if type(op_type) ~= "string" or type(name) ~= "string" then
    error("new_operation expects op type and name strings", 3)
  end
  local desc = lib.TF_NewOperation(graph:handle(), op_type, name)
  if desc == nil then error("TF_NewOperation failed", 3) end
  return setmetatable({ _desc = desc, _graph = graph, _name = name }, Description)
end

local function dtype_name(value)
  if type(value) == "table" and value.dtype then return value.dtype end
  return value or "float32"
end

function GraphBuilder.placeholder(graph, dtype, shape, name)
  local desc = GraphBuilder.new_operation(graph, "Placeholder", name)
  desc:set_type("dtype", dtype_name(dtype))
  if shape ~= nil then desc:set_shape("shape", shape) end
  return desc:finish()
end

function GraphBuilder.gradients(graph, outputs, inputs)
  local lib = capi.load()
  local function array(values)
    local result = ffi.new("TF_Output[?]", #values)
    for i, value in ipairs(values) do
      local output = resolve_output(graph, value)
      result[i - 1].oper = output.oper
      result[i - 1].index = output.index
    end
    return result
  end
  if #outputs == 0 or #inputs == 0 then
    error("gradients expects at least one output and one input", 3)
  end
  local y = array(outputs)
  local x = array(inputs)
  local dy = ffi.new("TF_Output[?]", #inputs)
  local status = capi.status()
  lib.TF_AddGradients(graph:handle(), y, #outputs, x, #inputs, nil, status, dy)
  capi.check(status, "gradients")
  local result = {}
  for i = 1, #inputs do
    result[i] = { oper = dy[i - 1].oper, index = dy[i - 1].index }
  end
  return result
end

local function binary(graph, op, a, b, name, dtype)
  local desc = GraphBuilder.new_operation(graph, op, name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype_name(dtype))
  return desc:finish()
end

function GraphBuilder.add(graph, a, b, name, dtype)
  return binary(graph, "AddV2", a, b, name, dtype)
end

function GraphBuilder.matmul(graph, a, b, name, dtype)
  return binary(graph, "MatMul", a, b, name, dtype)
end

function GraphBuilder.sub(graph, a, b, name, dtype)
  return binary(graph, "Sub", a, b, name, dtype)
end

function GraphBuilder.mul(graph, a, b, name, dtype)
  return binary(graph, "Mul", a, b, name, dtype)
end

function GraphBuilder.square(graph, a, name, dtype)
  local desc = GraphBuilder.new_operation(graph, "Square", name)
  desc:add_input(a)
  desc:set_type("T", dtype_name(dtype))
  return desc:finish()
end

return GraphBuilder
