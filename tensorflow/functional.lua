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
    _targets = {}, _target_order = {}, _closed = false,
  }, Functional)

  for _, spec in ipairs(options.inputs) do
    if type(spec.name) ~= "string" or spec.name == "" then
      error("Functional inputs need names", 2)
    end
    local placeholder = tf.placeholder(self._graph, spec.dtype or "float32",
      node_shape(spec.shape, "input " .. spec.name), spec.name)
    self._inputs[spec.name] = placeholder
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
    for name, target in pairs(self._targets) do
      local mse = tf.losses.mean_squared_error(self._graph, outputs[name],
        target, { axes = { 0, 1 }, name = "loss_" .. name })
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
  local history = { loss = {} }
  for epoch = 1, options.epochs or 1 do
    local values = self._session:run(feeds, { self._loss }, self._updates)
    history.loss[#history.loss + 1] = values[1]:totable()
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

function Functional:close()
  if not self._closed then
    self._session:close()
    self._graph:delete()
    self._closed = true
  end
end

return Functional
