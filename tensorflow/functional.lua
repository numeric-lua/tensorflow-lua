--
-- Functional, multi-input/multi-output graph model.
--
-- The builder receives a named table of input tensors and returns:
--   outputs   table of named output tensors
--   variables optional table of tf.variable handles created by the builder
-- The model owns the graph/session, target placeholders, loss, and optimizer.
-- It is deliberately lower-level than Sequential; callers can compose any
-- TensorFlow graph and keep full control of weights.
--
local Functional = {}
Functional.__index = Functional

local function copy_shape(shape)
  local result = {}
  for index = 1, #shape do result[index] = shape[index] end
  return result
end

local function node_shape(shape, name)
  if type(shape) ~= "table" or #shape == 0 then
    error("Functional " .. name .. " needs a non-empty shape", 3)
  end
  local full = { -1 }
  for index = 1, #shape do full[#full + 1] = shape[index] end
  return full
end

function Functional.new(options)
  if type(options) ~= "table" then
    error("Functional.new expects an options table", 2)
  end
  local tf = require("tensorflow")
  if type(options.inputs) ~= "table" or #options.inputs == 0 then
    error("Functional.new requires a non-empty inputs list", 2)
  end
  if type(options.build) ~= "function" then
    error("Functional.new requires a build(inputs) function", 2)
  end
  if type(options.targets) ~= "table" or next(options.targets) == nil then
    error("Functional.new requires target specifications", 2)
  end

  local self = setmetatable({
    _graph = tf.Graph.new(), _inputs = {}, _input_order = {},
    _targets = {}, _target_order = {}, _input_shapes = {}, _closed = false,
  }, Functional)

  for _, spec in ipairs(options.inputs) do
    if type(spec.name) ~= "string" or spec.name == "" then
      error("Functional inputs need names", 2)
    end
    local placeholder = tf.placeholder(self._graph, spec.dtype or "float32",
      node_shape(spec.shape, "input " .. spec.name), spec.name)
    self._inputs[spec.name] = placeholder
    self._input_shapes[spec.name] = copy_shape(spec.shape)
    self._input_order[#self._input_order + 1] = spec.name
  end

  local outputs, variables = options.build(self._inputs, self._graph)
  if type(outputs) ~= "table" or next(outputs) == nil then
    error("Functional builder must return a non-empty outputs table", 2)
  end
  self._outputs = outputs
  self._variables = variables or {}
  if type(self._variables) ~= "table" then
    error("Functional builder variables must be a table", 2)
  end

  for name, spec in pairs(options.targets) do
    if outputs[name] == nil then
      error("Functional target '" .. name .. "' has no output", 2)
    end
    local placeholder = tf.placeholder(self._graph, spec.dtype or "float32",
      node_shape(spec.shape, "target " .. name), "target_" .. name)
    self._targets[name] = placeholder
    self._target_order[#self._target_order + 1] = name
  end
  table.sort(self._target_order)

  if options.loss ~= nil then
    if type(options.loss) ~= "function" then
      error("Functional loss must be a function", 2)
    end
    self._loss = options.loss(self._graph, outputs, self._targets)
  else
    local total
    local weights = options.loss_weights or {}
    for name, target in pairs(self._targets) do
      local mse = tf.losses.mean_squared_error(self._graph, outputs[name],
        target, { axes = { 0, 1 }, name = "loss_" .. name })
      local weight = weights[name] or 1
      if weight ~= 1 then
        mse = tf.mul(self._graph, mse, tf.constant(self._graph, weight,
          "loss_weight_" .. name, "float32"), "loss_weighted_" .. name,
          "float32")
      end
      total = total == nil and mse or tf.add(self._graph, total, mse,
        "loss_sum_" .. name)
    end
    self._loss = total
  end
  if self._loss == nil then error("Functional loss builder returned nil", 2) end

  self._updates = {}
  if options.optimizer ~= nil and #self._variables > 0 then
    local optimizer = options.optimizer
    if type(optimizer) == "string" then optimizer = { type = optimizer } end
    local kind = optimizer.type or "sgd"
    local fn = kind == "sgd" and tf.train.gradient_descent or tf.train[kind]
    if fn == nil then error("unsupported Functional optimizer '" .. kind .. "'", 2) end
    local updates, _, state
    if kind == "sgd" or kind == "gradient_descent" then
      updates, _, state = tf.train.gradient_descent(self._graph, self._loss,
        self._variables, optimizer.learning_rate or optimizer.lr or 0.01,
        optimizer.name, optimizer)
    else
      updates, _, state = fn(self._graph, self._loss, self._variables, optimizer)
    end
    self._updates = updates or {}
    self._state = state
  end

  local initializers = {}
  for _, variable in ipairs(self._variables) do
    if variable.initializer ~= nil then
      initializers[#initializers + 1] = variable.initializer
    end
  end
  if self._state and self._state.initializers then
    for _, value in ipairs(self._state.initializers) do
      initializers[#initializers + 1] = value
    end
  end
  self._session = tf.Session.new(self._graph)
  self._session:run({}, {}, initializers)
  return self
end

local function feeds_for(self, values)
  local feeds = {}
  for _, name in ipairs(self._input_order) do
    local value = values[name]
    if value == nil then error("Functional input '" .. name .. "' is missing", 3) end
    feeds[self._inputs[name]] = value
  end
  return feeds
end

function Functional:predict(inputs)
  local tf = require("tensorflow")
  local feeds = feeds_for(self, inputs)
  local fetch, names = {}, {}
  for name, output in pairs(self._outputs) do
    fetch[#fetch + 1] = output
    names[#names + 1] = name
  end
  local values = self._session:run(feeds, fetch)
  local result = {}
  for index, name in ipairs(names) do result[name] = values[index] end
  return result
end

function Functional:fit(inputs, targets, options)
  options = options or {}
  local feeds = feeds_for(self, inputs)
  for _, name in ipairs(self._target_order) do
    if targets[name] == nil then
      error("Functional target '" .. name .. "' is missing", 2)
    end
    feeds[self._targets[name]] = targets[name]
  end
  local callbacks = options.callbacks or {}
  local history = { loss = {} }
  for _, callback in ipairs(callbacks) do
    if type(callback.on_train_begin) == "function" then
      callback.on_train_begin(history, self)
    end
  end
  for epoch = 1, options.epochs or 1 do
    local values = self._session:run(feeds, { self._loss }, self._updates)
    history.loss[#history.loss + 1] = values[1]:totable()
    local validation = options.validation_data
    if validation ~= nil then
      history.val_loss = history.val_loss or {}
      history.val_loss[#history.val_loss + 1] =
        self:evaluate(validation.inputs or validation[1],
          validation.targets or validation[2])
    end
    for _, callback in ipairs(callbacks) do
      if type(callback.on_epoch_end) == "function" then
        callback.on_epoch_end(history, epoch, self)
      end
    end
    if history.stop_training then break end
  end
  for _, callback in ipairs(callbacks) do
    if type(callback.on_train_end) == "function" then
      callback.on_train_end(history, self)
    end
  end
  return history
end

function Functional:evaluate(inputs, targets)
  local feeds = feeds_for(self, inputs)
  for _, name in ipairs(self._target_order) do
    if targets[name] == nil then
      error("Functional target '" .. name .. "' is missing", 2)
    end
    feeds[self._targets[name]] = targets[name]
  end
  return self._session:run(feeds, { self._loss })[1]:totable()
end

local function entries_for(variables)
  local entries = {}
  for index, variable in ipairs(variables) do
    entries[#entries + 1] = {
      name = variable.name or ("variable_" .. index), variable = variable,
    }
  end
  return entries
end

function Functional:get_weights()
  if self._session == nil then error("model is closed", 2) end
  local fetch = {}
  for index, variable in ipairs(self._variables) do fetch[index] = variable.read end
  return self._session:run({}, fetch)
end

function Functional:set_weights(weights)
  if type(weights) ~= "table" or #weights ~= #self._variables then
    error("set_weights expects one tensor per variable", 2)
  end
  local tf = require("tensorflow")
  self._set_counter = (self._set_counter or 0) + 1
  local targets = {}
  for index, variable in ipairs(self._variables) do
    local constant = tf.constant(self._graph, weights[index],
      "functional_set_value_" .. self._set_counter .. "_" .. index,
      variable.dtype)
    local desc = self._graph:new_operation("Assign",
      "functional_set_weight_" .. self._set_counter .. "_" .. index)
    desc:add_input(variable.ref)
    desc:add_input(constant)
    desc:set_type("T", variable.dtype)
    desc:set_bool("use_locking", true)
    desc:set_bool("validate_shape", true)
    targets[#targets + 1] = desc:finish()
  end
  self._session:run({}, {}, targets)
  return self
end

function Functional:count_params()
  local total = 0
  for _, variable in ipairs(self._variables) do
    local count = 1
    for _, dimension in ipairs(variable.shape or {}) do count = count * dimension end
    total = total + count
  end
  return total
end

function Functional:summary()
  local rows = { "Functional model", "Inputs:" }
  for _, name in ipairs(self._input_order) do
    rows[#rows + 1] = "  " .. name .. " " ..
      table.concat(self._input_shapes[name], "x")
  end
  rows[#rows + 1] = "Outputs: " .. table.concat((function()
    local names = {}
    for name in pairs(self._outputs) do names[#names + 1] = name end
    table.sort(names)
    return names
  end)(), ", ")
  rows[#rows + 1] = string.format("Variables: %d", #self._variables)
  rows[#rows + 1] = string.format("Total params: %d", self:count_params())
  local text = table.concat(rows, "\n")
  print(text)
  return text
end

function Functional:save(prefix)
  local tf = require("tensorflow")
  self._session:run({}, {}, { tf.save_v2(self._graph, prefix,
    entries_for(self._variables)) })
  return prefix
end

function Functional:load(prefix)
  local tf = require("tensorflow")
  self._session:run({}, {}, tf.restore_v2(self._graph, prefix,
    entries_for(self._variables)))
  return self
end

function Functional:export(path)
  local file, err = io.open(path, "wb")
  if file == nil then error("cannot open GraphDef: " .. tostring(err), 2) end
  local ok, write_err = pcall(function() file:write(self._graph:graph_def()) end)
  file:close()
  if not ok then error(write_err, 2) end
  return path
end

function Functional:close()
  if not self._closed then
    self._session:close()
    self._graph:delete()
    self._closed = true
  end
end

return Functional
