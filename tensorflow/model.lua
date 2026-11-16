--
-- Minimal Sequential model: Dense layers, a loss, an optimizer, and fit/predict.
--
local Sequential = {}
Sequential.__index = Sequential

local function copy_shape(shape)
  local result = {}
  for i, value in ipairs(shape) do result[i] = value end
  return result
end

local function batch_slice(data, first, last)
  local batch = {}
  for i = first, last do batch[#batch + 1] = data[i] end
  return batch
end

local function normalize_rows(data, width)
  if type(data) ~= "table" then
    error("model input must be a Lua table or Tensor", 3)
  end
  if #data == 0 then return data end
  if type(data[1]) ~= "table" and width ~= nil then
    local rows = {}
    for i, value in ipairs(data) do rows[i] = { value } end
    return rows
  end
  return data
end

function Sequential.new(options)
  options = options or {}
  if type(options.input_shape) ~= "table" then
    error("Sequential.new requires input_shape", 2)
  end
  return setmetatable({ _input_shape = copy_shape(options.input_shape),
    _layers = {}, _compiled = nil }, Sequential)
end

function Sequential:add(options)
  if type(options) ~= "table" or type(options.units) ~= "number" then
    error("add expects a layer definition with units", 2)
  end
  self._layers[#self._layers + 1] = options
  return self
end

function Sequential:compile(options)
  options = options or {}
  if options.optimizer == nil then options.optimizer = { type = "adam" } end
  if options.loss == nil then options.loss = "mse" end
  self._compiled = options
  return self
end

local function add_initializers(target, source)
  if type(source) ~= "table" then return end
  for _, value in ipairs(source) do target[#target + 1] = value end
end

function Sequential:_build()
  if self._graph ~= nil then return end
  local tf = require("tensorflow")
  if #self._layers == 0 then error("model has no layers", 3) end
  if self._compiled == nil then self:compile() end

  local graph = tf.Graph.new()
  local input_shape = { -1 }
  for _, dimension in ipairs(self._input_shape) do input_shape[#input_shape + 1] = dimension end
  local input = tf.placeholder(graph, "float32", input_shape, "input")
  local output = input
  local variables, initializers, layers = {}, {}, {}
  for i, definition in ipairs(self._layers) do
    local options = definition.options or {}
    options.input_size = options.input_size or
      (i == 1 and self._input_shape[#self._input_shape] or layers[i - 1].units)
    options.name = options.name or ("dense_" .. i)
    local layer = tf.layers.dense(graph, output, definition.units, options)
    layers[i] = layer
    for _, variable in ipairs(layer.variables) do variables[#variables + 1] = variable end
    add_initializers(initializers, layer.initializers)
    output = layer.output
  end
  local units = layers[#layers].units
  local target = tf.placeholder(graph, "float32", { -1, units }, "target")

  local loss_spec = self._compiled.loss
  local loss
  if type(loss_spec) == "function" then
    loss = loss_spec(graph, output, target)
  elseif loss_spec == "mse" or loss_spec == "mean_squared_error" then
    loss = tf.losses.mean_squared_error(graph, output, target, { axes = { 0, 1 } })
  elseif loss_spec == "softmax_cross_entropy" or loss_spec == "cce" then
    loss = tf.losses.softmax_cross_entropy(graph, output, target, { axes = { 0 } })
  else
    error("unsupported loss '" .. tostring(loss_spec) .. "'", 3)
  end

  local optimizer = self._compiled.optimizer
  if type(optimizer) == "string" then optimizer = { type = optimizer } end
  local optimizer_type = optimizer.type or "adam"
  local targets, state
  if optimizer_type == "adam" then
    local result = { tf.train.adam(graph, loss, variables, optimizer) }
    targets, state = result[1], result[3]
  elseif optimizer_type == "sgd" or optimizer_type == "gradient_descent" then
    targets = tf.train.gradient_descent(graph, loss, variables,
      optimizer.learning_rate or optimizer.lr or 0.01, optimizer.name)
  elseif optimizer_type == "rmsprop" then
    local result = { tf.train.rmsprop(graph, loss, variables, optimizer) }
    targets, state = result[1], result[3]
  elseif optimizer_type == "adagrad" then
    local result = { tf.train.adagrad(graph, loss, variables, optimizer) }
    targets, state = result[1], result[3]
  elseif optimizer_type == "momentum" then
    local result = { tf.train.momentum(graph, loss, variables, optimizer) }
    targets, state = result[1], result[3]
  else
    error("unsupported optimizer '" .. tostring(optimizer_type) .. "'", 3)
  end
  if type(state) == "table" then add_initializers(initializers, state.initializers) end

  local session = tf.Session.new(graph)
  session:run({}, {}, initializers)
  self._graph, self._session = graph, session
  self._input, self._target, self._loss, self._output = input, target, loss,
    output
  self._updates, self._variables, self._initializers = targets, variables,
    initializers
  self._layers_built = layers
  self._units = units
end

function Sequential:fit(x, y, options)
  options = options or {}
  if self._session == nil then self._graph = nil end
  self:_build()
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  y = normalize_rows(y, self._units)
  if #x ~= #y then error("x and y must have the same batch size", 2) end
  local epochs = options.epochs or 1
  local batch_size = options.batch_size or #x
  local history = { loss = {} }
  for _ = 1, epochs do
    local epoch_loss, batches = 0, 0
    for first = 1, #x, batch_size do
      local last = math.min(first + batch_size - 1, #x)
      local batch_x = batch_slice(x, first, last)
      local batch_y = batch_slice(y, first, last)
      local loss = self._session:run({ [self._input] = batch_x,
        [self._target] = batch_y }, { self._loss }, self._updates)[1]:totable()
      epoch_loss = epoch_loss + loss
      batches = batches + 1
    end
    history.loss[#history.loss + 1] = epoch_loss / batches
  end
  return history
end

function Sequential:predict(x)
  if self._session == nil then self._graph = nil end
  self:_build()
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  return self._session:run({ [self._input] = x }, { self._output })[1]
end

function Sequential:evaluate(x, y, options)
  options = options or {}
  if self._session == nil then self._graph = nil end
  self:_build()
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  y = normalize_rows(y, self._units)
  local batch_size = options.batch_size or #x
  local total, batches = 0, 0
  for first = 1, #x, batch_size do
    local last = math.min(first + batch_size - 1, #x)
    total = total + self._session:run({ [self._input] = batch_slice(x, first, last),
      [self._target] = batch_slice(y, first, last) }, { self._loss })[1]:totable()
    batches = batches + 1
  end
  return total / batches
end

function Sequential:save(path)
  if self._session == nil then error("model has no session to save", 2) end
  local entries = {}
  for _, variable in ipairs(self._variables) do
    entries[#entries + 1] = { name = variable.name, variable = variable }
  end
  return require("tensorflow").save_checkpoint(self._session, path, entries)
end

function Sequential:load(path)
  if self._session == nil then self._graph = nil end
  self:_build()
  local entries = {}
  for _, variable in ipairs(self._variables) do
    entries[#entries + 1] = { name = variable.name, variable = variable }
  end
  local targets = require("tensorflow").load_checkpoint(self._graph, path,
    entries)
  self._session:run({}, {}, targets)
  return self
end

function Sequential:close()
  if self._session ~= nil then self._session:close() end
  self._session = nil
  self._graph = nil
end

function Sequential:summary()
  self:_build()
  local rows = { "Sequential" }
  for i, layer in ipairs(self._layers_built) do
    rows[#rows + 1] = string.format("  %d: Dense(%d -> %d)", i,
      layer.input_size, layer.units)
  end
  return table.concat(rows, "\n")
end

return Sequential
