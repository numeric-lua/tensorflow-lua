--
-- Graph-mode variables and basic training helpers.
--
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
  local clipvalue = options.clipvalue or options.clip_value
  if clipvalue ~= nil then
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
  local beta2_power = Train.variable(graph, beta2, name .. "/beta2_power",
    { dtype = dtype })
  local initializers = { beta1_power.initializer, beta2_power.initializer }
  local weight_decay = options.weight_decay or options.wd or 0
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
    local decay_assign
    if weight_decay > 0 then
      local tensorflow = require("tensorflow")
      local factor = Train.constant(graph, 1 - learning_rate * weight_decay,
        name .. "/decay_" .. i, var_dtype)
      local decayed = tensorflow.mul(graph, variable.read or ref, factor,
        name .. "/decayed_" .. i, var_dtype)
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
    desc:set_bool("use_nesterov", false)
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
  return targets, gradients, { initializers = initializers, beta1_power =
    beta1_power, beta2_power = beta2_power }
end

function Train.adamw(graph, loss, variables, options)
  options = options or {}
  options.weight_decay = options.weight_decay or options.wd or 0.01
  return Train.adam(graph, loss, variables, options)
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
    dtypes[i] = variable.dtype or "float32"
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
    assign_desc:set_type("T", variable.dtype or "float32")
    assign_desc:set_bool("use_locking", true)
    targets[#targets + 1] = assign_desc:finish()
  end
  return targets
end

function Train.rmsprop(graph, loss, variables, options)
  options = options or {}
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
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local rho_op = Train.constant(graph, rho, name .. "/rho", dtype)
  local momentum_op = Train.constant(graph, momentum, name .. "/momentum", dtype)
  local epsilon_op = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local targets, initializers = {}, {}
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
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
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
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local rho_op = Train.constant(graph, rho, name .. "/rho", dtype)
  local momentum_op = Train.constant(graph, momentum, name .. "/momentum", dtype)
  local epsilon_op = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local targets, initializers = {}, {}
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
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
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
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local targets, initializers = {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local accum = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/accum_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    local desc = graph:new_operation("ApplyAdagrad", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(lr)
    desc:add_input(gradients[i])
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:set_bool("update_slots", true)
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
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
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local rho_op = Train.constant(graph, rho, name .. "/rho", dtype)
  local epsilon_op = Train.constant(graph, epsilon, name .. "/epsilon", dtype)
  local targets, initializers = {}, {}
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
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
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
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate",
    dtype)
  local l1_op = Train.constant(graph, l1, name .. "/l1", dtype)
  local l2_op = Train.constant(graph, l2, name .. "/l2", dtype)
  local targets, initializers = {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local accum = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/accum_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
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
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
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
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local l1_op = Train.constant(graph, l1, name .. "/l1", dtype)
  local l2_op = Train.constant(graph, l2, name .. "/l2", dtype)
  local power = Train.constant(graph, lr_power, name .. "/lr_power", dtype)
  local targets, initializers = {}, {}
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
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
end

function Train.momentum(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.01
  local momentum_value = options.momentum or 0.9
  local name = options.name or "Momentum"
  if type(variables) ~= "table" or #variables == 0 then
    error("momentum expects a non-empty variable list", 3)
  end
  local refs = {}
  for i, variable in ipairs(variables) do refs[i] = variable.ref or variable end
  local gradients = require("tensorflow").gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = Train.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local mom = Train.constant(graph, momentum_value, name .. "/momentum", dtype)
  local targets, initializers = {}, {}
  for i, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local accum = Train.variable(graph, zeros(variable.shape or { }),
      name .. "/accum_" .. i, { dtype = var_dtype })
    initializers[#initializers + 1] = accum.initializer
    local desc = graph:new_operation("ApplyMomentum", name .. "/update_" .. i)
    desc:add_input(ref)
    desc:add_input(accum.ref)
    desc:add_input(lr)
    desc:add_input(gradients[i])
    desc:add_input(mom)
    desc:set_type("T", var_dtype)
    desc:set_bool("use_locking", false)
    desc:set_bool("use_nesterov", false)
    targets[#targets + 1] = desc:finish()
  end
  return targets, gradients, { initializers = initializers }
end

function Train.adamw(graph, loss, variables, options)
  options = options or {}
  options.weight_decay = options.weight_decay or options.wd or 0.01
  return Train.adam(graph, loss, variables, options)
end

return Train
