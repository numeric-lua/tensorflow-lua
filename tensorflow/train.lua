--
-- Graph-mode variables and basic training helpers.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")

local Train = {}

local function dtype_name(value)
  if type(value) == "table" and value.dtype then return value.dtype end
  return value or "float32"
end

function Train.constant(graph, value, name, dtype)
  dtype = dtype_name(dtype)
  local tensor = getmetatable(value) == Tensor and value or
    Tensor.new(value, { dtype = dtype })
  local desc = graph:new_operation("Const", name)
  desc:set_tensor("value", tensor)
  desc:set_type("dtype", dtype)
  return desc:finish()
end

function Train.variable(graph, initial, name, options)
  options = options or {}
  local dtype = dtype_name(options.dtype or
    (getmetatable(initial) == Tensor and initial:dtype() or "float32"))
  local initial_tensor
  if getmetatable(initial) == Tensor then
    initial_tensor = initial
  else
    initial_tensor = Tensor.new(initial, { dtype = dtype })
  end
  local shape = initial_tensor:shape()

  local variable_desc = graph:new_operation("VariableV2", name)
  variable_desc:set_type("dtype", dtype)
  variable_desc:set_shape("shape", shape)
  if options.container then variable_desc:set_string("container", options.container) end
  if options.shared_name then variable_desc:set_string("shared_name", options.shared_name) end
  local variable = variable_desc:finish()

  local initializer_desc = graph:new_operation("Assign", name .. "/Assign")
  initializer_desc:add_input(variable)
  initializer_desc:add_input(Train.constant(graph, initial_tensor,
    name .. "/initial_value", dtype))
  initializer_desc:set_type("T", dtype)
  initializer_desc:set_bool("use_locking", true)
  initializer_desc:set_bool("validate_shape", true)
  local initializer = initializer_desc:finish()

  local read_desc = graph:new_operation("Identity", name .. "/read")
  read_desc:add_input(variable)
  read_desc:set_type("T", dtype)
  local read = read_desc:finish()

  return {
    ref = variable,
    read = read,
    initializer = initializer,
    dtype = dtype,
    shape = shape,
    name = name,
  }
end

function Train.gradient_descent(graph, loss, variables, learning_rate, name)
  name = name or "GradientDescent"
  if type(variables) ~= "table" or #variables == 0 then
    error("gradient_descent expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do
    refs[i] = variable.ref or variable
  end
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local lr_dtype = variables[1].dtype or variables[1].ref.dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    lr_dtype)
  local targets = {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local dtype = variable.dtype or "float32"
    local desc = graph:new_operation("ApplyGradientDescent",
      name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(lr)
    desc:add_input(gradients[i])
    desc:set_type("T", dtype)
    desc:set_bool("use_locking", false)
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, lr
end

return Train
