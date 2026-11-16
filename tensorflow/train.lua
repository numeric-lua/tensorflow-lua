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

local function zeros(shape, depth)
  depth = depth or 1
  if depth > #shape then return 0 end
  local result = {}
  for i = 1, shape[depth] do result[i] = zeros(shape, depth + 1) end
  return result
end

local function assign(graph, variable, value, name, control)
  local desc = graph:new_operation("Assign", name)
  desc:add_input(variable)
  desc:add_input(value)
  desc:set_type("T", variable.dtype or "float32")
  desc:set_bool("use_locking", true)
  if control then desc:add_control_input(control) end
  return desc:finish()
end

function Train.adam(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local name = options.name or "Adam"
  if type(variables) ~= "table" or #variables == 0 then
    error("adam expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local b1 = Train.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = Train.constant(graph, beta2, name .. "/beta2", dtype)
  local eps = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local beta1_power = Train.variable(graph, beta1, name .. "/beta1_power",
    { dtype = dtype })
  local beta2_power = Train.variable(graph, beta2, name .. "/beta2_power",
    { dtype = dtype })
  local initializers = { beta1_power.initializer, beta2_power.initializer }
  local targets, apply_ops = {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local m = Train.variable(graph, zeros(shape), name .. "/m_" .. i,
      { dtype = var_dtype })
    local v = Train.variable(graph, zeros(shape), name .. "/v_" .. i,
      { dtype = var_dtype })
    initializers[#initializers + 1] = m.initializer
    initializers[#initializers + 1] = v.initializer
    local desc = graph:new_operation("ApplyAdam", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(m.ref)
    desc:add_input(v.ref)
    desc:add_input(beta1_power.read)
    desc:add_input(beta2_power.read)
    desc:add_input(lr)
    desc:add_input(b1)
    desc:add_input(b2)
    desc:add_input(eps)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:set_bool("use_nesterov", false)
    local apply = desc:finish()
    apply_ops[#apply_ops + 1] = apply
    targets[#targets + 1] = apply
  end
  local next_b1 = require("tensorflow").mul(graph, beta1_power.read, b1,
    name .. "/next_beta1")
  local next_b2 = require("tensorflow").mul(graph, beta2_power.read, b2,
    name .. "/next_beta2")
  for _, apply in ipairs(apply_ops) do
    targets[#targets + 1] = assign(graph, beta1_power.ref, next_b1,
      name .. "/assign_beta1", apply)
    targets[#targets + 1] = assign(graph, beta2_power.ref, next_b2,
      name .. "/assign_beta2", apply)
    break
  end
  return targets, gradients, { initializers = initializers, beta1_power =
    beta1_power, beta2_power = beta2_power }
end

return Train
