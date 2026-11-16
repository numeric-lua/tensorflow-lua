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

local function normalize_labels(data)
  if type(data) ~= "table" then error("labels must be a Lua table", 3) end
  if #data == 0 or type(data[1]) ~= "table" then return data end
  local flat = {}
  for i, row in ipairs(data) do
    if type(row) ~= "table" or #row ~= 1 then
      error("sparse labels must be a flat list or single-column rows", 3)
    end
    flat[i] = row[1]
  end
  return flat
end

local function is_dataset(value)
  return type(value) == "table" and type(value.data) == "function" and
    value._x ~= nil
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
  if type(options) ~= "table" or type(options.units) ~= "number" or
      options.units < 1 then
    error("add expects a layer definition with positive units", 2)
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
  local loss_spec = self._compiled.loss
  local sparse = loss_spec == "sparse_softmax_cross_entropy" or
    loss_spec == "sparse_ce" or
    loss_spec == "sparse_categorical_cross_entropy"
  local target = tf.placeholder(graph, sparse and "int32" or "float32",
    sparse and { -1 } or { -1, units }, "target")

  local loss
  if type(loss_spec) == "function" then
    loss = loss_spec(graph, output, target)
  elseif loss_spec == "mse" or loss_spec == "mean_squared_error" then
    loss = tf.losses.mean_squared_error(graph, output, target, { axes = { 0, 1 } })
  elseif loss_spec == "softmax_cross_entropy" or loss_spec == "cce" then
    loss = tf.losses.softmax_cross_entropy(graph, output, target, { axes = { 0 } })
  elseif loss_spec == "mae" or loss_spec == "mean_absolute_error" then
    loss = tf.losses.mean_absolute_error(graph, output, target,
      { axes = { 0, 1 } })
  elseif loss_spec == "binary_cross_entropy" or loss_spec == "bce" then
    loss = tf.losses.binary_cross_entropy(graph, output, target,
      { axes = { 0, 1 } })
  elseif sparse then
    local one_hot = tf.one_hot(graph, target, units, { name = "target_one_hot" })
    loss = tf.losses.softmax_cross_entropy(graph, output, one_hot,
      { axes = { 0 } })
  else
    error("unsupported loss '" .. tostring(loss_spec) .. "'", 3)
  end

  local metric_outputs, metric_names = {}, {}
  for i, name in ipairs(self._compiled.metrics or {}) do
    if name == "accuracy" then
      local predictions = tf.argmax(graph, output, 1, "metric/prediction",
        "float32", "int64")
      local labels, dtype
      if sparse then
        predictions = tf.cast(graph, predictions, "int64", "int32",
          "metric/prediction_i32")
        labels, dtype = target, "int32"
      else
        labels = tf.argmax(graph, target, 1, "metric/label", "float32",
          "int64")
        dtype = "int64"
      end
      metric_outputs[i] = tf.metrics.accuracy(graph, predictions, labels,
        { dtype = dtype, name = "metric/accuracy" })
      metric_names[i] = "accuracy"
    elseif name == "binary_accuracy" then
      local probabilities = tf.nn.sigmoid(graph, output,
        "metric/binary_sigmoid")
      metric_outputs[i] = tf.metrics.binary_accuracy(graph, probabilities,
        target, { axes = { 0, 1 }, name = "metric/binary_accuracy" })
      metric_names[i] = "binary_accuracy"
    else
      error("unsupported metric '" .. tostring(name) .. "'", 3)
    end
  end

  local optimizer = self._compiled.optimizer
  if type(optimizer) == "string" then optimizer = { type = optimizer } end
  local optimizer_type = optimizer.type or "adam"
  local targets, state
  if optimizer_type == "adam" then
    local result = { tf.train.adam(graph, loss, variables, optimizer) }
    targets, state = result[1], result[3]
  elseif optimizer_type == "adamw" then
    local result = { tf.train.adamw(graph, loss, variables, optimizer) }
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
  elseif optimizer_type == "adadelta" then
    local result = { tf.train.adadelta(graph, loss, variables, optimizer) }
    targets, state = result[1], result[3]
  elseif optimizer_type == "ftrl" then
    local result = { tf.train.ftrl(graph, loss, variables, optimizer) }
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
  self._sparse = sparse
  self._metric_names, self._metric_outputs = metric_names, metric_outputs
end

function Sequential:fit(x, y, options)
  if is_dataset(x) and options == nil and type(y) == "table" and
      (y.epochs ~= nil or y.batch_size ~= nil or y.callbacks ~= nil or
        y.validation_data ~= nil or y.shuffle ~= nil) then
    options, y = y, nil
  end
  options = options or {}
  if is_dataset(x) then
    local dataset_x, dataset_y, batch_size, shuffle, seed = x:data()
    x, y = dataset_x, y or dataset_y
    options.batch_size = options.batch_size or batch_size
    if options.shuffle == nil then options.shuffle = shuffle end
    if options.seed == nil then options.seed = seed end
  end
  if self._session == nil then self._graph = nil end
  self:_build()
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  y = self._sparse and normalize_labels(y) or normalize_rows(y, self._units)
  if #x == 0 then error("fit requires a non-empty dataset", 2) end
  if #x ~= #y then error("x and y must have the same batch size", 2) end
  if options.shuffle then
    if options.seed ~= nil then math.randomseed(options.seed) end
    local order = {}
    for i = 1, #x do order[i] = i end
    for i = #order, 2, -1 do
      local j = math.random(i)
      order[i], order[j] = order[j], order[i]
    end
    local shuffled_x, shuffled_y = {}, {}
    for i, index in ipairs(order) do
      shuffled_x[i] = x[index]
      shuffled_y[i] = y[index]
    end
    x, y = shuffled_x, shuffled_y
  end
  local epochs = options.epochs or 1
  local batch_size = options.batch_size or #x
  local callbacks = options.callbacks or {}
  local validation = options.validation_data
  local history = { loss = {} }
  if validation ~= nil then
    history.val_loss = {}
    for _, name in ipairs(self._metric_names or {}) do
      history["val_" .. name] = {}
    end
  end
  for _, name in ipairs(self._metric_names or {}) do history[name] = {} end
  for _, callback in ipairs(callbacks) do
    if type(callback.on_train_begin) == "function" then
      callback.on_train_begin(history, self)
    end
  end
  for epoch = 1, epochs do
    local epoch_loss, batches = 0, 0
    local metric_sums = {}
    for first = 1, #x, batch_size do
      local last = math.min(first + batch_size - 1, #x)
      local batch_x = batch_slice(x, first, last)
      local batch_y = batch_slice(y, first, last)
      local fetch = { self._loss }
      for _, output in ipairs(self._metric_outputs or {}) do
        fetch[#fetch + 1] = output
      end
      local values = self._session:run({ [self._input] = batch_x,
        [self._target] = batch_y }, fetch, self._updates)
      epoch_loss = epoch_loss + values[1]:totable()
      for i = 1, #(self._metric_outputs or {}) do
        metric_sums[i] = (metric_sums[i] or 0) + values[i + 1]:totable()
      end
      batches = batches + 1
    end
    history.loss[#history.loss + 1] = epoch_loss / batches
    for i, name in ipairs(self._metric_names or {}) do
      history[name][#history[name] + 1] = metric_sums[i] / batches
    end
    if validation ~= nil then
      local val_loss, val_metrics
      if is_dataset(validation) then
        val_loss, val_metrics = self:evaluate(validation, nil,
          { batch_size = batch_size })
      else
        val_loss, val_metrics = self:evaluate(validation[1],
          validation[2], { batch_size = batch_size })
      end
      history.val_loss[#history.val_loss + 1] = val_loss
      for name, value in pairs(val_metrics or {}) do
        local target_history = history["val_" .. name]
        if target_history ~= nil then target_history[#target_history + 1] = value end
      end
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

function Sequential:fit_dataset(dataset, options)
  return self:fit(dataset, nil, options)
end

function Sequential:evaluate_dataset(dataset, options)
  return self:evaluate(dataset, nil, options)
end

function Sequential:predict_dataset(dataset)
  return self:predict(dataset)
end

function Sequential:predict(x)
  if is_dataset(x) then x = x:data() end
  if self._session == nil then self._graph = nil end
  self:_build()
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  if #x == 0 then error("predict requires non-empty input", 2) end
  return self._session:run({ [self._input] = x }, { self._output })[1]
end

function Sequential:evaluate(x, y, options)
  options = options or {}
  if is_dataset(x) then
    local dataset_x, dataset_y = x:data()
    x, y = dataset_x, y or dataset_y
  end
  if self._session == nil then self._graph = nil end
  self:_build()
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  y = self._sparse and normalize_labels(y) or normalize_rows(y, self._units)
  local batch_size = options.batch_size or #x
  local total, batches = 0, 0
  local metric_sums = {}
  for first = 1, #x, batch_size do
    local last = math.min(first + batch_size - 1, #x)
    local fetch = { self._loss }
    for _, output in ipairs(self._metric_outputs or {}) do
      fetch[#fetch + 1] = output
    end
    local values = self._session:run({ [self._input] = batch_slice(x, first, last),
      [self._target] = batch_slice(y, first, last) }, fetch)
    total = total + values[1]:totable()
    for i = 1, #(self._metric_outputs or {}) do
      metric_sums[i] = (metric_sums[i] or 0) + values[i + 1]:totable()
    end
    batches = batches + 1
  end
  local metrics = {}
  for i, name in ipairs(self._metric_names or {}) do
    metrics[name] = metric_sums[i] / batches
  end
  return total / batches, metrics
end

function Sequential:export(path)
  if self._session == nil then self._graph = nil end
  self:_build()
  local file, err = io.open(path, "wb")
  if file == nil then error("cannot open GraphDef: " .. tostring(err), 2) end
  local ok, write_err = pcall(function()
    file:write(self._graph:graph_def())
  end)
  file:close()
  if not ok then error(write_err, 2) end
  return path
end

function Sequential:save(path)
  if self._session == nil then error("model has no session to save", 2) end
  local entries = {}
  for _, variable in ipairs(self._variables) do
    entries[#entries + 1] = { name = variable.name, variable = variable }
  end
  return require("tensorflow").save_checkpoint(self._session, path, entries)
end

function Sequential:save_native(prefix)
  if self._session == nil then error("model has no session to save", 2) end
  local tf = require("tensorflow")
  local entries = {}
  for _, variable in ipairs(self._variables) do
    entries[#entries + 1] = { name = variable.name, variable = variable }
  end
  self._session:run({}, {}, { tf.save_v2(self._graph, prefix, entries) })
  return prefix
end

function Sequential:load_native(prefix)
  if self._session == nil then self._graph = nil end
  self:_build()
  local tf = require("tensorflow")
  local entries = {}
  for _, variable in ipairs(self._variables) do
    entries[#entries + 1] = { name = variable.name, variable = variable }
  end
  self._session:run({}, {}, tf.restore_v2(self._graph, prefix, entries))
  return self
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
