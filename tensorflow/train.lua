--
-- Graph-mode variables and basic training helpers.
--
local Tensor = require("tensorflow.tensor")
local Ops = require("tensorflow.ops")

local Train = {}

local function dtype_name(value)
  if type(value) == "table" and value.dtype then return value.dtype end
  return value or "float32"
end

function Train.constant(graph, value, name, dtype)
  if type(value) == "table" and value.oper ~= nil then return value end
  dtype = dtype_name(dtype)
  local tensor = getmetatable(value) == Tensor and value or
    Tensor.new(value, { dtype = dtype })
  local desc = graph:new_operation("Const", name)
  desc:set_tensor("value", tensor)
  desc:set_type("dtype", dtype)
  local output = desc:finish()
  output.dtype = dtype
  return output
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
  variable.dtype = dtype

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
  read.dtype = dtype

  return {
    ref = variable,
    read = read,
    initializer = initializer,
    dtype = dtype,
    shape = shape,
    name = name,
  }
end

local function maximum(graph, a, b, name, dtype)
  local desc = graph:new_operation("Maximum", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  return desc:finish()
end

local function multiply(graph, a, b, name, dtype)
  local desc = graph:new_operation("Mul", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  local output = desc:finish()
  output.dtype = dtype
  return output
end

local function add(graph, a, b, name, dtype)
  local desc = graph:new_operation("AddV2", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  local output = desc:finish()
  output.dtype = dtype
  return output
end

local function subtract(graph, a, b, name, dtype)
  local desc = graph:new_operation("Sub", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  local output = desc:finish()
  output.dtype = dtype
  return output
end

local function weight_decay_value(options)
  local weight_decay = options.weight_decay or options.wd or 0
  if type(weight_decay) ~= "number" or weight_decay < 0 then
    error("weight_decay must be a non-negative number", 3)
  end
  return weight_decay
end

local function weight_decay_assign(graph, variable, learning_rate, weight_decay,
    name, dtype)
  if weight_decay == 0 then return nil end
  local tf = require("tensorflow")
  local factor
  if type(learning_rate) == "table" and learning_rate.oper ~= nil then
    local one = Train.constant(graph, 1, name .. "/decay_one", dtype)
    local decay = Train.constant(graph, weight_decay, name .. "/decay_weight",
      dtype)
    factor = tf.sub(graph, one,
      tf.mul(graph, learning_rate, decay, name .. "/decay_scale", dtype),
      name .. "/decay_factor", dtype)
  else
    factor = Train.constant(graph, 1 - learning_rate * weight_decay,
      name .. "/decay_factor", dtype)
  end
  local ref = variable.ref or variable
  local read = variable.read or ref
  local var_dtype = type(variable) == "table" and variable.dtype or dtype
  local decayed = multiply(graph, read, factor, name .. "/decayed",
    var_dtype or dtype)
  local desc = graph:new_operation("Assign", name .. "/weight_decay")
  desc:add_input(ref)
  desc:add_input(decayed)
  desc:set_type("T", var_dtype or dtype)
  desc:set_bool("use_locking", true)
  return desc:finish()
end

local function check_clip_norm(clip_norm)
  if type(clip_norm) ~= "number" or clip_norm <= 0 then
    error("clipnorm/clip_norm must be a positive number", 3)
  end
end

local function square(graph, input, name, dtype)
  local desc = graph:new_operation("Square", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  local output = desc:finish()
  output.dtype = dtype
  return output
end

local function gradient_list(gradients, label)
  if type(gradients) ~= "table" or #gradients == 0 then
    error(label .. " expects a non-empty gradient list", 3)
  end
  for index, gradient in ipairs(gradients) do
    if type(gradient) ~= "table" or gradient.oper == nil then
      error(label .. " gradient " .. tostring(index) ..
        " must be a graph output", 3)
    end
  end
end

local function gradient_dtype(gradients, dtype)
  if dtype ~= nil then return dtype_name(dtype) end
  for _, gradient in ipairs(gradients) do
    if type(gradient.dtype) == "string" then return gradient.dtype end
  end
  return "float32"
end

local function squared_norm(graph, gradient, name, dtype)
  local flat = Ops.reshape(graph, gradient, { -1 }, name .. "/flatten", dtype)
  local squared = square(graph, flat, name .. "/square", dtype)
  return Ops.reduce_sum(graph, squared, { 0 }, name .. "/sum", dtype)
end

function Train.clip_by_norm(graph, gradient, clip_norm, name, dtype)
  check_clip_norm(clip_norm)
  gradient_list({ gradient }, "clip_by_norm")
  name = name or "clip_by_norm"
  local value_dtype = gradient_dtype({ gradient }, dtype)
  local norm = Ops.sqrt(graph, squared_norm(graph, gradient, name, value_dtype),
    name .. "/norm", value_dtype)
  local limit = Train.constant(graph, clip_norm, name .. "/limit", value_dtype)
  local denominator = maximum(graph, norm, limit, name .. "/denominator",
    value_dtype)
  local scale = Ops.div(graph, limit, denominator, name .. "/scale", value_dtype)
  return multiply(graph, gradient, scale, name .. "/clipped", value_dtype), norm
end

function Train.clip_by_global_norm(graph, gradients, clip_norm, name, dtype)
  check_clip_norm(clip_norm)
  gradient_list(gradients, "clip_by_global_norm")
  name = name or "clip_by_global_norm"
  local value_dtype = gradient_dtype(gradients, dtype)
  local total
  for index, gradient in ipairs(gradients) do
    local sum = squared_norm(graph, gradient, name .. "/grad_" .. index,
      value_dtype)
    if total == nil then
      total = sum
    else
      total = add(graph, total, sum, name .. "/total_" .. index, value_dtype)
    end
  end
  local norm = Ops.sqrt(graph, total, name .. "/norm", value_dtype)
  local limit = Train.constant(graph, clip_norm, name .. "/limit", value_dtype)
  local denominator = maximum(graph, norm, limit, name .. "/denominator",
    value_dtype)
  local scale = Ops.div(graph, limit, denominator, name .. "/scale", value_dtype)
  local clipped = {}
  for index, gradient in ipairs(gradients) do
    clipped[index] = multiply(graph, gradient, scale,
      name .. "/clipped_" .. index, value_dtype)
  end
  return clipped, norm
end

local function train_gradients(graph, loss, refs, options, name)
  local tf = require("tensorflow")
  local gradients = tf.gradients(graph, { loss }, refs)
  local clipnorm = options and (options.clipnorm or options.clip_norm)
  if clipnorm ~= nil then
    local dtype = refs[1].dtype or "float32"
    if type(dtype) ~= "string" then dtype = "float32" end
    gradients = Train.clip_by_global_norm(graph, gradients, clipnorm,
      name .. "/clipnorm", dtype)
  end
  return gradients
end

local function apply_clipvalue(graph, gradients, variables, clipvalue, name,
    dtype)
  if clipvalue == nil then return gradients end
  if type(clipvalue) ~= "number" or clipvalue <= 0 then
    error("clipvalue must be a positive number", 3)
  end
  local tf = require("tensorflow")
  local clipped = {}
  for i, gradient in ipairs(gradients) do
    local variable = variables[i]
    local var_dtype = type(variable) == "table" and variable.dtype or dtype
    clipped[i] = tf.clip_by_value(graph, gradient, -clipvalue, clipvalue,
      name .. "/clipvalue_" .. i, var_dtype or "float32")
  end
  return clipped
end

function Train.gradient_descent(graph, loss, variables, learning_rate, name,
    options)
  name = name or "GradientDescent"
  options = options or {}
  if type(variables) ~= "table" or #variables == 0 then
    error("gradient_descent expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do
    refs[i] = variable.ref or variable
  end
  local gradients = options.gradients or train_gradients(graph, loss, refs,
    options, name)
  local clipvalue = options.clipvalue or options.clip_value
  if clipvalue ~= nil then
    if type(clipvalue) ~= "number" or clipvalue <= 0 then
      error("clipvalue must be a positive number", 3)
    end
    local tf = require("tensorflow")
    for i, gradient in ipairs(gradients) do
      local dtype = variables[i].dtype or "float32"
      gradients[i] = tf.clip_by_value(graph, gradient, -clipvalue, clipvalue,
        name .. "/clipvalue_" .. i, dtype)
    end
  end
  local lr_dtype = variables[1].dtype or variables[1].ref.dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    lr_dtype)
  local weight_decay = weight_decay_value(options)
  local targets = {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local dtype = variable.dtype or "float32"
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, dtype)
    local desc = graph:new_operation("ApplyGradientDescent",
      name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(lr)
    desc:add_input(gradients[i])
    desc:set_type("T", dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, lr
end

function Train.sgd(graph, loss, variables, options)
  options = options or {}
  local momentum = options.momentum
  local nesterov = options.nesterov or options.use_nesterov
  if (momentum ~= nil and momentum ~= 0) or nesterov then
    return Train.momentum(graph, loss, variables, options)
  end
  local learning_rate = options.learning_rate or options.lr or 0.01
  local name = options.name or "SGD"
  return Train.gradient_descent(graph, loss, variables, learning_rate, name,
    options)
end

local function zeros(shape, depth)
  depth = depth or 1
  if depth > #shape then return 0 end
  local result = {}
  for i = 1, shape[depth] do result[i] = zeros(shape, depth + 1) end
  return result
end

local function assign(graph, variable, value, name, control)
  local ref = variable.ref or variable
  local dtype = type(variable) == "table" and variable.dtype or "float32"
  local desc = graph:new_operation("Assign", name)
  desc:add_input(ref)
  desc:add_input(value)
  desc:set_type("T", dtype)
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
  local nesterov = options.nesterov
  if nesterov == nil then nesterov = options.use_nesterov end
  if nesterov ~= nil and type(nesterov) ~= "boolean" then
    error("adam nesterov must be a boolean", 3)
  end
  local clipvalue = options.clipvalue or options.clip_value
  if clipvalue ~= nil and (type(clipvalue) ~= "number" or clipvalue <= 0) then
    error("clipvalue must be a positive number", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("adam expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  local dtype = variables[1].dtype or "float32"
  if clipvalue ~= nil then
    for i, gradient in ipairs(gradients) do
      gradients[i] = require("tensorflow").clip_by_value(graph, gradient,
        -clipvalue, clipvalue, name .. "/clip_" .. i,
        variables[i].dtype or dtype)
    end
  end
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
  local state_variables = { beta1_power, beta2_power }
  local weight_decay = options.weight_decay or options.wd or 0
  local decay_factor
  if weight_decay > 0 then
    if type(learning_rate) == "table" and learning_rate.oper ~= nil then
      local tensorflow = require("tensorflow")
      local one = Train.constant(graph, 1, name .. "/decay_one", dtype)
      local decay = Train.constant(graph, weight_decay,
        name .. "/decay_weight", dtype)
      decay_factor = tensorflow.sub(graph, one,
        tensorflow.mul(graph, lr, decay, name .. "/decay_scale", dtype),
        name .. "/decay_factor", dtype)
    else
      decay_factor = Train.constant(graph, 1 - learning_rate * weight_decay,
        name .. "/decay_factor", dtype)
    end
  end
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
    state_variables[#state_variables + 1] = m
    state_variables[#state_variables + 1] = v
    local decay_assign
    if weight_decay > 0 then
      local tensorflow = require("tensorflow")
      local decayed = tensorflow.mul(graph, variable.read or ref,
        decay_factor, name .. "/decayed_" .. i, var_dtype)
      decay_assign = assign(graph, ref, decayed,
        name .. "/weight_decay_" .. i)
    end
    local desc = graph:new_operation("ApplyAdam", name .. "/update_" .. i)
    if decay_assign ~= nil then desc:add_control_input(decay_assign) end
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
    desc:set_bool("use_nesterov", nesterov == true)
    local apply = desc:finish()
    apply_ops[#apply_ops + 1] = apply
    targets[#targets + 1] = apply
  end
  local next_b1 = require("tensorflow").mul(graph, beta1_power.read, b1,
    name .. "/next_beta1")
  local next_b2 = require("tensorflow").mul(graph, beta2_power.read, b2,
    name .. "/next_beta2")
  local apply = apply_ops[1]
  if apply ~= nil then
    targets[#targets + 1] = assign(graph, beta1_power.ref, next_b1,
      name .. "/assign_beta1", apply)
    targets[#targets + 1] = assign(graph, beta2_power.ref, next_b2,
      name .. "/assign_beta2", apply)
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables, beta1_power = beta1_power,
    beta2_power = beta2_power }
end

function Train.adamw(graph, loss, variables, options)
  options = options or {}
  options.weight_decay = options.weight_decay or options.wd or 0.01
  return Train.adam(graph, loss, variables, options)
end

function Train.adamax(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local name = options.name or "AdaMax"
  if type(variables) ~= "table" or #variables == 0 then
    error("adamax expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs,
    options, name)
  local dtype = variables[1].dtype or "float32"
  local clipvalue = options.clipvalue or options.clip_value
  if clipvalue ~= nil then
    if type(clipvalue) ~= "number" or clipvalue <= 0 then
      error("clipvalue must be a positive number", 3)
    end
    for i, gradient in ipairs(gradients) do
      gradients[i] = require("tensorflow").clip_by_value(graph, gradient,
        -clipvalue, clipvalue, name .. "/clip_" .. i, dtype)
    end
  end
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local b1 = Train.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = Train.constant(graph, beta2, name .. "/beta2", dtype)
  local eps = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local beta1_power = Train.variable(graph, beta1, name .. "/beta1_power",
    { dtype = dtype })
  local weight_decay = weight_decay_value(options)
  local initializers = { beta1_power.initializer }
  local state_variables = { beta1_power }
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
    state_variables[#state_variables + 1] = m
    state_variables[#state_variables + 1] = v
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyAdaMax", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(m.ref)
    desc:add_input(v.ref)
    desc:add_input(beta1_power.read)
    desc:add_input(lr)
    desc:add_input(b1)
    desc:add_input(b2)
    desc:add_input(eps)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    local apply = desc:finish()
    apply_ops[#apply_ops + 1] = apply
    targets[#targets + 1] = apply
  end
  local next_b1 = require("tensorflow").mul(graph, beta1_power.read, b1,
    name .. "/next_beta1")
  local apply = apply_ops[1]
  if apply ~= nil then
    targets[#targets + 1] = assign(graph, beta1_power.ref, next_b1,
      name .. "/assign_beta1", apply)
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables, beta1_power = beta1_power }
end

local function string_vector(graph, values, name)
  return Train.constant(graph, values, name, "string")
end

function Train.save_v2(graph, prefix, variables, name)
  if type(variables) ~= "table" or #variables == 0 then
    error("save_v2 expects a non-empty variable list", 3)
  end
  local names, slices, dtypes, refs = {}, {}, {}, {}
  for i, entry in ipairs(variables) do
    local variable = entry.variable or entry
    names[i] = entry.name or variable.name
    if names[i] == nil then error("save_v2 variable is missing a name", 3) end
    slices[i] = ""
    dtypes[i] = variable.dtype or "float32"
    refs[i] = variable.ref or variable
  end
  local desc = graph:new_operation("SaveV2", name or "save")
  desc:add_input(Train.constant(graph, prefix, (name or "save") .. "/prefix",
    "string"))
  desc:add_input(string_vector(graph, names, (name or "save") .. "/names"))
  desc:add_input(string_vector(graph, slices, (name or "save") .. "/slices"))
  desc:add_inputs(refs)
  desc:set_type_list("dtypes", dtypes)
  return desc:finish()
end

function Train.restore_v2(graph, prefix, variables, name)
  if type(variables) ~= "table" or #variables == 0 then
    error("restore_v2 expects a non-empty variable list", 3)
  end
  local names, slices, dtypes = {}, {}, {}
  for i, entry in ipairs(variables) do
    local variable = entry.variable or entry
    names[i] = entry.name or variable.name
    if names[i] == nil then error("restore_v2 variable is missing a name", 3) end
    slices[i] = ""
    dtypes[i] = entry.dtype or variable.dtype or "float32"
  end
  local prefix_op = Train.constant(graph, prefix, (name or "restore") .. "/prefix",
    "string")
  local desc = graph:new_operation("RestoreV2", (name or "restore") .. "/read")
  desc:add_input(prefix_op)
  desc:add_input(string_vector(graph, names, (name or "restore") .. "/names"))
  desc:add_input(string_vector(graph, slices, (name or "restore") .. "/slices"))
  desc:set_type_list("dtypes", dtypes)
  local restore = desc:finish()
  local targets = {}
  for i, entry in ipairs(variables) do
    local variable = entry.variable or entry
    local value = { oper = restore.oper, index = i - 1,
      name = (name or "restore") .. "/read:" .. (i - 1) }
    local assign_desc = graph:new_operation("Assign",
      (name or "restore") .. "/assign_" .. i)
    assign_desc:add_input(variable.ref or variable)
    assign_desc:add_input(value)
    assign_desc:set_type("T", entry.dtype or variable.dtype or "float32")
    assign_desc:set_bool("use_locking", true)
    targets[#targets + 1] = assign_desc:finish()
  end
  return targets
end

function Train.rmsprop(graph, loss, variables, options)
  options = options or {}
  if options.centered ~= nil and type(options.centered) ~= "boolean" then
    error("rmsprop centered must be a boolean", 3)
  end
  if options.centered then
    return Train.centered_rmsprop(graph, loss, variables, options)
  end
  local learning_rate = options.learning_rate or options.lr or 0.001
  local rho = options.rho or 0.9
  local momentum = options.momentum or 0
  local epsilon = options.epsilon or 1e-10
  local name = options.name or "RMSProp"
  if type(variables) ~= "table" or #variables == 0 then
    error("rmsprop expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  gradients = apply_clipvalue(graph, gradients, variables,
    options.clipvalue or options.clip_value, name,
    variables[1].dtype or "float32")
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local rho_op = Train.constant(graph, rho, name .. "/rho", dtype)
  local momentum_op = Train.constant(graph, momentum, name .. "/momentum", dtype)
  local epsilon_op = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local ms = Train.variable(graph, zeros(shape), name .. "/ms_" .. i,
      { dtype = var_dtype })
    local mom = Train.variable(graph, zeros(shape), name .. "/mom_" .. i,
      { dtype = var_dtype })
    initializers[#initializers + 1] = ms.initializer
    initializers[#initializers + 1] = mom.initializer
    state_variables[#state_variables + 1] = ms
    state_variables[#state_variables + 1] = mom
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyRMSProp", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(ms.ref)
    desc:add_input(mom.ref)
    desc:add_input(lr)
    desc:add_input(rho_op)
    desc:add_input(momentum_op)
    desc:add_input(epsilon_op)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.centered_rmsprop(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local rho = options.rho or 0.9
  local momentum = options.momentum or 0
  local epsilon = options.epsilon or 1e-10
  local name = options.name or "CenteredRMSProp"
  if type(variables) ~= "table" or #variables == 0 then
    error("centered_rmsprop expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  gradients = apply_clipvalue(graph, gradients, variables,
    options.clipvalue or options.clip_value, name,
    variables[1].dtype or "float32")
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local rho_op = Train.constant(graph, rho, name .. "/rho", dtype)
  local momentum_op = Train.constant(graph, momentum, name .. "/momentum", dtype)
  local epsilon_op = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local mg = Train.variable(graph, zeros(shape), name .. "/mg_" .. i,
      { dtype = var_dtype })
    local ms = Train.variable(graph, zeros(shape), name .. "/ms_" .. i,
      { dtype = var_dtype })
    local mom = Train.variable(graph, zeros(shape), name .. "/mom_" .. i,
      { dtype = var_dtype })
    initializers[#initializers + 1] = mg.initializer
    initializers[#initializers + 1] = ms.initializer
    initializers[#initializers + 1] = mom.initializer
    state_variables[#state_variables + 1] = mg
    state_variables[#state_variables + 1] = ms
    state_variables[#state_variables + 1] = mom
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyCenteredRMSProp",
      name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(mg.ref)
    desc:add_input(ms.ref)
    desc:add_input(mom.ref)
    desc:add_input(lr)
    desc:add_input(rho_op)
    desc:add_input(momentum_op)
    desc:add_input(epsilon_op)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.adagrad(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.01
  local name = options.name or "Adagrad"
  if type(variables) ~= "table" or #variables == 0 then
    error("adagrad expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  gradients = apply_clipvalue(graph, gradients, variables,
    options.clipvalue or options.clip_value, name,
    variables[1].dtype or "float32")
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local accum = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/accum_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    state_variables[#state_variables + 1] = accum
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyAdagrad", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(lr)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:set_bool("update_slots", true)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.adadelta(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 1.0
  local rho = options.rho or 0.95
  local epsilon = options.epsilon or 1e-8
  local name = options.name or "Adadelta"
  if type(variables) ~= "table" or #variables == 0 then
    error("adadelta expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  gradients = apply_clipvalue(graph, gradients, variables,
    options.clipvalue or options.clip_value, name,
    variables[1].dtype or "float32")
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local rho_op = Train.constant(graph, rho, name .. "/rho", dtype)
  local epsilon_op = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local accum = Train.variable(graph, zeros(shape), name .. "/accum_" .. i,
      { dtype = var_dtype })
    local update = Train.variable(graph, zeros(shape),
      name .. "/update_slot_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    initializers[#initializers + 1] = update.initializer
    state_variables[#state_variables + 1] = accum
    state_variables[#state_variables + 1] = update
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyAdadelta", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(update.ref)
    desc:add_input(lr)
    desc:add_input(rho_op)
    desc:add_input(epsilon_op)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.proximal_adagrad(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.01
  local l1 = options.l1 or 0
  local l2 = options.l2 or 0
  local name = options.name or "ProximalAdagrad"
  if type(variables) ~= "table" or #variables == 0 then
    error("proximal_adagrad expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  gradients = apply_clipvalue(graph, gradients, variables,
    options.clipvalue or options.clip_value, name,
    variables[1].dtype or "float32")
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local l1_op = Train.constant(graph, l1, name .. "/l1", dtype)
  local l2_op = Train.constant(graph, l2, name .. "/l2", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local accum = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/accum_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    state_variables[#state_variables + 1] = accum
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyProximalAdagrad",
      name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(lr)
    desc:add_input(l1_op)
    desc:add_input(l2_op)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.ftrl(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local l1 = options.l1 or 0
  local l2 = options.l2 or 0
  local lr_power = options.lr_power or -0.5
  local name = options.name or "Ftrl"
  if type(variables) ~= "table" or #variables == 0 then
    error("ftrl expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  gradients = apply_clipvalue(graph, gradients, variables,
    options.clipvalue or options.clip_value, name,
    variables[1].dtype or "float32")
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local l1_op = Train.constant(graph, l1, name .. "/l1", dtype)
  local l2_op = Train.constant(graph, l2, name .. "/l2", dtype)
  local power = Train.constant(graph, lr_power, name .. "/lr_power", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local accum = Train.variable(graph, zeros(shape), name .. "/accum_" .. i,
      { dtype = var_dtype })
    local linear = Train.variable(graph, zeros(shape), name .. "/linear_" .. i,
      { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    initializers[#initializers + 1] = linear.initializer
    state_variables[#state_variables + 1] = accum
    state_variables[#state_variables + 1] = linear
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyFtrl", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(linear.ref)
    desc:add_input(gradients[i])
    desc:add_input(lr)
    desc:add_input(l1_op)
    desc:add_input(l2_op)
    desc:add_input(power)
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.momentum(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.01
  local momentum_value = options.momentum or 0.9
  local name = options.name or "Momentum"
  local nesterov = options.nesterov
  if nesterov == nil then nesterov = options.use_nesterov end
  if nesterov ~= nil and type(nesterov) ~= "boolean" then
    error("momentum nesterov must be a boolean", 3)
  end
  local clipvalue = options.clipvalue or options.clip_value
  if clipvalue ~= nil and (type(clipvalue) ~= "number" or clipvalue <= 0) then
    error("clipvalue must be a positive number", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("momentum expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or train_gradients(graph, loss, refs, options, name)
  local dtype = variables[1].dtype or "float32"
  if clipvalue ~= nil then
    for i, gradient in ipairs(gradients) do
      gradients[i] = require("tensorflow").clip_by_value(graph, gradient,
        -clipvalue, clipvalue, name .. "/clip_" .. i,
        variables[i].dtype or dtype)
    end
  end
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local mom = Train.constant(graph, momentum_value, name .. "/momentum", dtype)
  local weight_decay = weight_decay_value(options)
  local targets, initializers, state_variables = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local accum = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/accum_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    state_variables[#state_variables + 1] = accum
    local decay = weight_decay_assign(graph, variable, lr, weight_decay,
      name .. "/wd_" .. i, var_dtype)
    local desc = graph:new_operation("ApplyMomentum", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(lr)
    desc:add_input(gradients[i])
    desc:add_input(mom)
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:set_bool("use_nesterov", nesterov == true)
    if decay ~= nil then desc:add_control_input(decay) end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.exponential_moving_average(graph, variables, decay, options)
  options = options or {}
  if type(decay) ~= "number" or decay < 0 or decay > 1 then
    error("exponential_moving_average decay must be between 0 and 1", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("exponential_moving_average expects a non-empty variable list", 3)
  end
  local name = options.name or "ExponentialMovingAverage"
  local averages, initializers, updates = {}, {}, {}
  for i, variable in ipairs(variables) do
    local var_dtype = variable.dtype or "float32"
    local average = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/average_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = average.initializer
    initializers[#initializers + 1] = assign(graph, average,
      variable.read or variable.ref, name .. "/sync_" .. i,
      average.initializer)
    local decay_op = Train.constant(graph, decay, name .. "/decay_" .. i,
      var_dtype)
    local one_minus = Train.constant(graph, 1 - decay,
      name .. "/one_minus_decay_" .. i, var_dtype)
    local weighted_average = multiply(graph, average.read, decay_op,
      name .. "/weighted_average_" .. i, var_dtype)
    local weighted_value = multiply(graph, variable.read or variable.ref,
      one_minus, name .. "/weighted_value_" .. i, var_dtype)
    local next_average = add(graph, weighted_average, weighted_value,
      name .. "/next_" .. i, var_dtype)
    updates[#updates + 1] = assign(graph, average, next_average,
      name .. "/update_" .. i)
    averages[#averages + 1] = average
  end
  return { averages = averages, initializers = initializers,
    updates = updates }
end

function Train.gradient_accumulation(graph, variables, gradients, steps, options)
  options = options or {}
  if type(steps) ~= "number" or steps % 1 ~= 0 or steps < 1 then
    error("gradient_accumulation steps must be a positive integer", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("gradient_accumulation expects a non-empty variable list", 3)
  end
  if type(gradients) ~= "table" or #gradients ~= #variables then
    error("gradient_accumulation gradients must match variables", 3)
  end
  local name = options.name or "GradientAccumulation"
  local accumulators, initializers, accumulate, reset = {}, {}, {}, {}
  local averages = {}
  for i, variable in ipairs(variables) do
    local gradient = gradients[i]
    if type(gradient) ~= "table" or gradient.oper == nil then
      error("gradient_accumulation gradient " .. i .. " must be a graph output",
        3)
    end
    local dtype = variable.dtype or "float32"
    local shape = variable.shape or { }
    local accumulator = Train.variable(graph, zeros(shape),
      name .. "/accumulator_" .. i, { dtype = dtype })
    initializers[#initializers + 1] = accumulator.initializer
    local add_op = graph:new_operation("AssignAdd", name .. "/add_" .. i)
    add_op:add_input(accumulator.ref)
    add_op:add_input(gradient)
    add_op:set_type("T", dtype)
    add_op:set_bool("use_locking", false)
    accumulate[#accumulate + 1] = add_op:finish()
    local zero = Train.constant(graph, zeros(shape), name .. "/zero_" .. i,
      dtype)
    reset[#reset + 1] = assign(graph, accumulator, zero,
      name .. "/reset_" .. i)
    local average = Ops.div(graph, accumulator.read,
      Train.constant(graph, steps, name .. "/steps_" .. i, dtype),
      name .. "/average_" .. i, dtype)
    average.dtype = dtype
    averages[#averages + 1] = average
    accumulators[#accumulators + 1] = accumulator
  end
  return {
    variables = accumulators,
    initializers = initializers,
    accumulate = accumulate,
    reset = reset,
    gradients = averages,
  }
end

function Train.amsgrad(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local name = options.name or "AMSGrad"
  if beta1 < 0 or beta1 > 1 then
    error("amsgrad beta1 must be between 0 and 1", 3)
  end
  if beta2 < 0 or beta2 > 1 then
    error("amsgrad beta2 must be between 0 and 1", 3)
  end
  if epsilon <= 0 then
    error("amsgrad epsilon must be positive", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("amsgrad expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or
    train_gradients(graph, loss, refs, options, name)
  if #gradients ~= #variables then
    error("amsgrad gradients must match variables", 3)
  end
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local b1 = Train.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = Train.constant(graph, beta2, name .. "/beta2", dtype)
  local one_minus_b1 = Train.constant(graph, 1 - beta1,
    name .. "/one_minus_beta1", dtype)
  local one_minus_b2 = Train.constant(graph, 1 - beta2,
    name .. "/one_minus_beta2", dtype)
  local eps = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local initializers, state_variables, targets = {}, {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local m = Train.variable(graph, zeros(shape), name .. "/m_" .. i,
      { dtype = var_dtype })
    local v = Train.variable(graph, zeros(shape), name .. "/v_" .. i,
      { dtype = var_dtype })
    local vhat = Train.variable(graph, zeros(shape), name .. "/vhat_" .. i,
      { dtype = var_dtype })
    initializers[#initializers + 1] = m.initializer
    initializers[#initializers + 1] = v.initializer
    initializers[#initializers + 1] = vhat.initializer
    state_variables[#state_variables + 1] = m
    state_variables[#state_variables + 1] = v
    state_variables[#state_variables + 1] = vhat
    local gradient = gradients[i]
    local m_next = add(graph, multiply(graph, m.read, b1,
      name .. "/m_scaled_" .. i, var_dtype), multiply(graph, gradient,
      one_minus_b1, name .. "/m_input_" .. i, var_dtype),
      name .. "/m_next_" .. i, var_dtype)
    local v_next = add(graph, multiply(graph, v.read, b2,
      name .. "/v_scaled_" .. i, var_dtype), multiply(graph,
      square(graph, gradient, name .. "/gradient_squared_" .. i, var_dtype),
      one_minus_b2, name .. "/v_input_" .. i, var_dtype),
      name .. "/v_next_" .. i, var_dtype)
    local vhat_next = maximum(graph, vhat.read, v_next,
      name .. "/vhat_next_" .. i, var_dtype)
    vhat_next.dtype = var_dtype
    local m_assign = assign(graph, m, m_next, name .. "/m_assign_" .. i)
    local v_assign = assign(graph, v, v_next, name .. "/v_assign_" .. i)
    local vhat_assign = assign(graph, vhat, vhat_next,
      name .. "/vhat_assign_" .. i)
    local denominator = add(graph, Ops.sqrt(graph, vhat_next,
      name .. "/sqrt_vhat_" .. i, var_dtype), eps,
      name .. "/denominator_" .. i, var_dtype)
    local step = Ops.div(graph, m_next, denominator,
      name .. "/step_" .. i, var_dtype)
    step.dtype = var_dtype
    local scaled = multiply(graph, lr, step, name .. "/scaled_" .. i,
      var_dtype)
    local next_value = subtract(graph, variable.read or ref, scaled,
      name .. "/next_" .. i, var_dtype)
    next_value.dtype = var_dtype
    local desc = graph:new_operation("Assign", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(next_value)
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:add_control_input(m_assign)
    desc:add_control_input(v_assign)
    desc:add_control_input(vhat_assign)
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.nadam(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local name = options.name or "NAdam"
  if beta1 < 0 or beta1 > 1 then
    error("nadam beta1 must be between 0 and 1", 3)
  end
  if beta2 < 0 or beta2 > 1 then
    error("nadam beta2 must be between 0 and 1", 3)
  end
  if epsilon <= 0 then
    error("nadam epsilon must be positive", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("nadam expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or
    train_gradients(graph, loss, refs, options, name)
  if #gradients ~= #variables then
    error("nadam gradients must match variables", 3)
  end
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local b1 = Train.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = Train.constant(graph, beta2, name .. "/beta2", dtype)
  local one = Train.constant(graph, 1, name .. "/one", dtype)
  local one_minus_b1 = Train.constant(graph, 1 - beta1,
    name .. "/one_minus_beta1", dtype)
  local one_minus_b2 = Train.constant(graph, 1 - beta2,
    name .. "/one_minus_beta2", dtype)
  local eps = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local beta1_power = Train.variable(graph, beta1, name .. "/beta1_power",
    { dtype = dtype })
  local beta2_power = Train.variable(graph, beta2, name .. "/beta2_power",
    { dtype = dtype })
  local initializers = { beta1_power.initializer, beta2_power.initializer }
  local state_variables = { beta1_power, beta2_power }
  local targets, variable_assigns = {}, {}
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
    state_variables[#state_variables + 1] = m
    state_variables[#state_variables + 1] = v
    local gradient = gradients[i]
    local m_next = add(graph, multiply(graph, m.read, b1,
      name .. "/m_scaled_" .. i, var_dtype), multiply(graph, gradient,
      one_minus_b1, name .. "/m_input_" .. i, var_dtype),
      name .. "/m_next_" .. i, var_dtype)
    local v_next = add(graph, multiply(graph, v.read, b2,
      name .. "/v_scaled_" .. i, var_dtype), multiply(graph,
      square(graph, gradient, name .. "/gradient_squared_" .. i, var_dtype),
      one_minus_b2, name .. "/v_input_" .. i, var_dtype),
      name .. "/v_next_" .. i, var_dtype)
    local one_minus_b1_power = subtract(graph, one, beta1_power.read,
      name .. "/one_minus_beta1_power_" .. i, var_dtype)
    local one_minus_b2_power = subtract(graph, one, beta2_power.read,
      name .. "/one_minus_beta2_power_" .. i, var_dtype)
    local m_hat = Ops.div(graph, m_next, one_minus_b1_power,
      name .. "/m_hat_" .. i, var_dtype)
    m_hat.dtype = var_dtype
    local v_hat = Ops.div(graph, v_next, one_minus_b2_power,
      name .. "/v_hat_" .. i, var_dtype)
    v_hat.dtype = var_dtype
    local gradient_hat = Ops.div(graph, gradient, one_minus_b1_power,
      name .. "/gradient_hat_" .. i, var_dtype)
    gradient_hat.dtype = var_dtype
    local m_nesterov = add(graph, multiply(graph, m_hat, b1,
      name .. "/m_nesterov_scaled_" .. i, var_dtype), multiply(graph,
      gradient_hat, one_minus_b1, name .. "/m_nesterov_input_" .. i,
      var_dtype), name .. "/m_nesterov_" .. i, var_dtype)
    local denominator = add(graph, Ops.sqrt(graph, v_hat,
      name .. "/sqrt_v_hat_" .. i, var_dtype), eps,
      name .. "/denominator_" .. i, var_dtype)
    local step = Ops.div(graph, m_nesterov, denominator,
      name .. "/step_" .. i, var_dtype)
    step.dtype = var_dtype
    local scaled = multiply(graph, lr, step, name .. "/scaled_" .. i,
      var_dtype)
    local next_value = subtract(graph, variable.read or ref, scaled,
      name .. "/next_" .. i, var_dtype)
    local m_assign = assign(graph, m, m_next, name .. "/m_assign_" .. i)
    local v_assign = assign(graph, v, v_next, name .. "/v_assign_" .. i)
    local desc = graph:new_operation("Assign", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(next_value)
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:add_control_input(m_assign)
    desc:add_control_input(v_assign)
    local update = desc:finish()
    targets[#targets + 1] = update
    variable_assigns[#variable_assigns + 1] = update
  end
  local b1_power_next = multiply(graph, beta1_power.read, b1,
    name .. "/beta1_power_next", dtype)
  local b2_power_next = multiply(graph, beta2_power.read, b2,
    name .. "/beta2_power_next", dtype)
  local b1_power_assign = graph:new_operation("Assign",
    name .. "/beta1_power_assign")
  b1_power_assign:add_input(beta1_power.ref)
  b1_power_assign:add_input(b1_power_next)
  b1_power_assign:set_type("T", dtype)
  b1_power_assign:set_bool("use_locking", false)
  for _, update in ipairs(variable_assigns) do
    b1_power_assign:add_control_input(update)
  end
  local b2_power_assign = graph:new_operation("Assign",
    name .. "/beta2_power_assign")
  b2_power_assign:add_input(beta2_power.ref)
  b2_power_assign:add_input(b2_power_next)
  b2_power_assign:set_type("T", dtype)
  b2_power_assign:set_bool("use_locking", false)
  for _, update in ipairs(variable_assigns) do
    b2_power_assign:add_control_input(update)
  end
  targets[#targets + 1] = b1_power_assign:finish()
  targets[#targets + 1] = b2_power_assign:finish()
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Train.radam(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local name = options.name or "RAdam"
  if beta1 < 0 or beta1 > 1 then
    error("radam beta1 must be between 0 and 1", 3)
  end
  if beta2 < 0 or beta2 > 1 then
    error("radam beta2 must be between 0 and 1", 3)
  end
  if epsilon <= 0 then
    error("radam epsilon must be positive", 3)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("radam expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = options.gradients or
    train_gradients(graph, loss, refs, options, name)
  if #gradients ~= #variables then
    error("radam gradients must match variables", 3)
  end
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local b1 = Train.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = Train.constant(graph, beta2, name .. "/beta2", dtype)
  local one = Train.constant(graph, 1, name .. "/one", dtype)
  local two = Train.constant(graph, 2, name .. "/two", dtype)
  local four = Train.constant(graph, 4, name .. "/four", dtype)
  local five = Train.constant(graph, 5, name .. "/five", dtype)
  local one_minus_b1 = Train.constant(graph, 1 - beta1,
    name .. "/one_minus_beta1", dtype)
  local one_minus_b2 = Train.constant(graph, 1 - beta2,
    name .. "/one_minus_beta2", dtype)
  local eps = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local rho_inf_value = 2 / (1 - beta2) - 1
  local rho_inf = Train.constant(graph, rho_inf_value, name .. "/rho_inf",
    dtype)
  local rho_inf_minus_4 = Train.constant(graph, rho_inf_value - 4,
    name .. "/rho_inf_minus_4", dtype)
  local rho_inf_minus_2 = Train.constant(graph, rho_inf_value - 2,
    name .. "/rho_inf_minus_2", dtype)
  local beta1_power = Train.variable(graph, beta1, name .. "/beta1_power",
    { dtype = dtype })
  local beta2_power = Train.variable(graph, beta2, name .. "/beta2_power",
    { dtype = dtype })
  local step_count = Train.variable(graph, 0, name .. "/step",
    { dtype = dtype })
  local initializers = { beta1_power.initializer, beta2_power.initializer,
    step_count.initializer }
  local state_variables = { beta1_power, beta2_power, step_count }
  local targets, variable_assigns = {}, {}
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
    state_variables[#state_variables + 1] = m
    state_variables[#state_variables + 1] = v
    local gradient = gradients[i]
    local m_next = add(graph, multiply(graph, m.read, b1,
      name .. "/m_scaled_" .. i, var_dtype), multiply(graph, gradient,
      one_minus_b1, name .. "/m_input_" .. i, var_dtype),
      name .. "/m_next_" .. i, var_dtype)
    local v_next = add(graph, multiply(graph, v.read, b2,
      name .. "/v_scaled_" .. i, var_dtype), multiply(graph,
      square(graph, gradient, name .. "/gradient_squared_" .. i, var_dtype),
      one_minus_b2, name .. "/v_input_" .. i, var_dtype),
      name .. "/v_next_" .. i, var_dtype)
    local one_minus_b1_power = subtract(graph, one, beta1_power.read,
      name .. "/one_minus_beta1_power_" .. i, var_dtype)
    local one_minus_b2_power = subtract(graph, one, beta2_power.read,
      name .. "/one_minus_beta2_power_" .. i, var_dtype)
    local m_hat = Ops.div(graph, m_next, one_minus_b1_power,
      name .. "/m_hat_" .. i, var_dtype)
    m_hat.dtype = var_dtype
    local v_hat = Ops.div(graph, v_next, one_minus_b2_power,
      name .. "/v_hat_" .. i, var_dtype)
    v_hat.dtype = var_dtype
    local step_next = add(graph, step_count.read, one,
      name .. "/step_next_" .. i, var_dtype)
    local rho_numerator = multiply(graph, multiply(graph, two, step_next,
      name .. "/rho_step_" .. i, var_dtype), beta2_power.read,
      name .. "/rho_numerator_" .. i, var_dtype)
    local rho_ratio = Ops.div(graph, rho_numerator, one_minus_b2_power,
      name .. "/rho_ratio_" .. i, var_dtype)
    rho_ratio.dtype = var_dtype
    local rho_t = subtract(graph, rho_inf, rho_ratio,
      name .. "/rho_t_" .. i, var_dtype)
    local rectifiable = Ops.greater(graph, rho_t, five,
      name .. "/rectifiable_" .. i)
    local rho_safe = maximum(graph, rho_t, five,
      name .. "/rho_safe_" .. i, var_dtype)
    rho_safe.dtype = var_dtype
    local rho_minus_4 = subtract(graph, rho_safe, four,
      name .. "/rho_minus_4_" .. i, var_dtype)
    local rho_minus_2 = subtract(graph, rho_safe, two,
      name .. "/rho_minus_2_" .. i, var_dtype)
    local rect_numerator = multiply(graph,
      multiply(graph, rho_minus_4, rho_minus_2,
        name .. "/rect_numerator_base_" .. i, var_dtype), rho_inf,
      name .. "/rect_numerator_" .. i, var_dtype)
    local rect_denominator = multiply(graph,
      multiply(graph, rho_inf_minus_4, rho_inf_minus_2,
        name .. "/rect_denominator_base_" .. i, var_dtype), rho_safe,
      name .. "/rect_denominator_" .. i, var_dtype)
    local rectification = Ops.sqrt(graph,
      Ops.div(graph, rect_numerator, rect_denominator,
        name .. "/rect_ratio_" .. i, var_dtype),
      name .. "/rectification_" .. i, var_dtype)
    rectification.dtype = var_dtype
    local adaptive = Ops.div(graph, m_hat, add(graph,
      Ops.sqrt(graph, v_hat, name .. "/sqrt_v_hat_" .. i, var_dtype), eps,
      name .. "/denominator_" .. i, var_dtype),
      name .. "/adaptive_" .. i, var_dtype)
    adaptive.dtype = var_dtype
    local rectified = multiply(graph, rectification, adaptive,
      name .. "/rectified_" .. i, var_dtype)
    local step = Ops.where(graph, rectifiable, rectified, m_hat,
      name .. "/step_" .. i, var_dtype)
    step.dtype = var_dtype
    local scaled = multiply(graph, lr, step, name .. "/scaled_" .. i,
      var_dtype)
    local next_value = subtract(graph, variable.read or ref, scaled,
      name .. "/next_" .. i, var_dtype)
    local m_assign = assign(graph, m, m_next, name .. "/m_assign_" .. i)
    local v_assign = assign(graph, v, v_next, name .. "/v_assign_" .. i)
    local desc = graph:new_operation("Assign", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(next_value)
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:add_control_input(m_assign)
    desc:add_control_input(v_assign)
    local update = desc:finish()
    targets[#targets + 1] = update
    variable_assigns[#variable_assigns + 1] = update
  end
  local state_updates = {
    { variable = beta1_power,
      value = multiply(graph, beta1_power.read, b1,
        name .. "/beta1_power_next", dtype) },
    { variable = beta2_power,
      value = multiply(graph, beta2_power.read, b2,
        name .. "/beta2_power_next", dtype) },
    { variable = step_count,
      value = add(graph, step_count.read, one, name .. "/step_next", dtype) },
  }
  for index, state in ipairs(state_updates) do
    local desc = graph:new_operation("Assign",
      name .. "/state_update_" .. index)
    desc:add_input(state.variable.ref)
    desc:add_input(state.value)
    desc:set_type("T", dtype)
    desc:set_bool("use_locking", false)
    for _, update in ipairs(variable_assigns) do
      desc:add_control_input(update)
    end
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

local optimizer_functions = {
  adam = function(graph, loss, variables, options)
    return Train.adam(graph, loss, variables, options)
  end,
  adamw = function(graph, loss, variables, options)
    return Train.adamw(graph, loss, variables, options)
  end,
  amsgrad = function(graph, loss, variables, options)
    return Train.amsgrad(graph, loss, variables, options)
  end,
  nadam = function(graph, loss, variables, options)
    return Train.nadam(graph, loss, variables, options)
  end,
  radam = function(graph, loss, variables, options)
    return Train.radam(graph, loss, variables, options)
  end,
  adamax = function(graph, loss, variables, options)
    return Train.adamax(graph, loss, variables, options)
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

function Train.optimizer(kind, graph, loss, variables, options)
  if type(kind) ~= "string" or kind == "" then
    error("optimizer kind must be a non-empty string", 3)
  end
  local key = string.lower(kind)
  if key == "sgd" or key == "gradient_descent" then
    return Train.sgd(graph, loss, variables, options)
  end
  local factory = optimizer_functions[key]
  if factory == nil then
    error("unknown optimizer '" .. kind .. "'", 3)
  end
  return factory(graph, loss, variables, options or {})
end

function Train.apply_gradients(kind, graph, variables, gradients, options)
  if type(gradients) ~= "table" or #gradients == 0 then
    error("apply_gradients expects a non-empty gradient list", 3)
  end
  if type(variables) ~= "table" or #variables ~= #gradients then
    error("apply_gradients gradients must match variables", 3)
  end
  for i, gradient in ipairs(gradients) do
    if type(gradient) ~= "table" or gradient.oper == nil then
      error("apply_gradients gradient " .. i .. " must be a graph output", 3)
    end
  end
  local opts = {}
  for key, value in pairs(options or {}) do opts[key] = value end
  opts.gradients = gradients
  return Train.optimizer(kind, graph, nil, variables, opts)
end

return Train
