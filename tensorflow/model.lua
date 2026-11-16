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
    _input_dtype = options.input_dtype, _layers = {}, _compiled = nil }, Sequential)
end

function Sequential:add(options)
  if type(options) ~= "table" then
    error("add expects a layer definition table", 2)
  end
  if options.type == "flatten" or options.type == "embedding" or
      options.type == "dropout" or options.type == "conv2d" or
      options.type == "conv1d" or options.type == "max_pool2d" or
      options.type == "avg_pool2d" or
      options.type == "global_avg_pool2d" or options.type == "batchnorm" or
      options.type == "batch_normalization" then
    if options.type == "embedding" and #self._layers == 0 and
        self._input_dtype == nil then
      self._input_dtype = "int32"
    end
    self._layers[#self._layers + 1] = options
    return self
  end
  if type(options.units) ~= "number" or options.units < 1 then
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
  local input = tf.placeholder(graph, self._input_dtype or "float32",
    input_shape, "input")
  local training = tf.placeholder(graph, "bool", { }, "training")
  local output = input
  local variables, initializers, layers = {}, {}, {}
  local batchnorm_updates, batchnorm_state = {}, {}
  local feature_size = self._input_shape[#self._input_shape]
  local spatial = copy_shape(self._input_shape)
  for i, definition in ipairs(self._layers) do
    local options = definition.options or {}
    options.name = options.name or ((definition.type or "dense") .. "_" .. i)
    if definition.type == "batchnorm" or
        definition.type == "batch_normalization" then
      local axes, channels
      if #spatial == 3 then
        axes = { 0, 1, 2 }
        channels = spatial[3]
      elseif #spatial == 2 then
        axes = { 0, 1 }
        channels = spatial[2]
      elseif #spatial == 1 then
        axes = { 0 }
        channels = spatial[1]
      else
        error("batchnorm requires a 2-D or NHWC input", 3)
      end
      local layer = tf.layers.batch_normalization(graph, output, channels, {
        training = training, axes = axes, name = options.name,
        epsilon = options.epsilon, momentum = options.momentum,
        dtype = options.dtype,
      })
      layers[i] = { output = layer.output, input_size = feature_size,
        units = feature_size, kind = "batchnorm" }
      for _, variable in ipairs(layer.variables) do
        variables[#variables + 1] = variable
      end
      add_initializers(initializers, layer.initializers)
      for _, update in ipairs(layer.updates or {}) do
        batchnorm_updates[#batchnorm_updates + 1] = update
      end
      batchnorm_state[#batchnorm_state + 1] = layer.moving_mean
      batchnorm_state[#batchnorm_state + 1] = layer.moving_variance
      output = layer.output
    elseif definition.type == "conv1d" then
      if #spatial ~= 2 then
        error("conv1d layers require input_shape = { steps, channels }", 3)
      end
      local kernel = definition.kernel_size
      local stride = definition.stride or 1
      local dilation = definition.dilation or 1
      local padding = string.upper(options.padding or definition.padding or "SAME")
      local layer = tf.layers.conv1d(graph, output, definition.filters,
        kernel, {
          in_channels = spatial[2], stride = stride, padding = padding,
          dilation = dilation, use_bias = options.use_bias,
          activation = options.activation, dtype = options.dtype,
          name = options.name,
        })
      local effective = (kernel - 1) * dilation + 1
      local steps
      if padding == "SAME" then
        steps = math.ceil(spatial[1] / stride)
      else
        steps = math.floor((spatial[1] - effective) / stride) + 1
      end
      spatial = { steps, definition.filters }
      feature_size = steps * definition.filters
      layers[i] = { output = layer.output, input_size = feature_size,
        units = definition.filters, kind = "conv1d",
        filters = definition.filters, kernel_size = kernel }
      for _, variable in ipairs(layer.variables) do
        variables[#variables + 1] = variable
      end
      add_initializers(initializers, layer.initializers)
      output = layer.output
    elseif definition.type == "conv2d" then
      if #spatial ~= 3 then
        error("conv2d layers require input_shape = { height, width, channels }", 3)
      end
      local kernel_h, kernel_w = definition.kernel_size,
        definition.kernel_size
      if type(definition.kernel_size) == "table" then
        kernel_h, kernel_w = definition.kernel_size[1], definition.kernel_size[2]
      end
      local stride_h, stride_w = definition.stride or 1, definition.stride or 1
      if type(definition.stride) == "table" then
        stride_h, stride_w = definition.stride[1], definition.stride[2]
      end
      local dilation_h, dilation_w = 1, 1
      if type(definition.dilation) == "table" then
        dilation_h, dilation_w = definition.dilation[1], definition.dilation[2]
      elseif definition.dilation ~= nil then
        dilation_h, dilation_w = definition.dilation, definition.dilation
      end
      local padding = string.upper(options.padding or definition.padding or "SAME")
      local conv_options = {
        in_channels = spatial[3], stride = { stride_h, stride_w },
        padding = padding, dilation = { dilation_h, dilation_w },
        use_bias = options.use_bias, activation = options.activation,
        dtype = options.dtype, name = options.name,
      }
      local layer = tf.layers.conv2d(graph, output, definition.filters,
        { kernel_h, kernel_w }, conv_options)
      local function output_dim(size, kernel, stride, dilation)
        local effective = (kernel - 1) * dilation + 1
        if padding == "SAME" then return math.ceil(size / stride) end
        return math.floor((size - effective) / stride) + 1
      end
      spatial = { output_dim(spatial[1], kernel_h, stride_h, dilation_h),
        output_dim(spatial[2], kernel_w, stride_w, dilation_w),
        definition.filters }
      feature_size = spatial[1] * spatial[2] * spatial[3]
      layers[i] = { output = layer.output, input_size = feature_size,
        units = definition.filters, kind = "conv2d",
        filters = definition.filters, kernel_size = { kernel_h, kernel_w } }
      for _, variable in ipairs(layer.variables) do
        variables[#variables + 1] = variable
      end
      add_initializers(initializers, layer.initializers)
      output = layer.output
    elseif definition.type == "max_pool2d" then
      if #spatial ~= 3 then
        error("max_pool2d layers require an NHWC input", 3)
      end
      local kernel_h, kernel_w = definition.kernel_size,
        definition.kernel_size
      if type(definition.kernel_size) == "table" then
        kernel_h, kernel_w = definition.kernel_size[1], definition.kernel_size[2]
      end
      local stride_h, stride_w = definition.stride or kernel_h,
        definition.stride or kernel_w
      if type(definition.stride) == "table" then
        stride_h, stride_w = definition.stride[1], definition.stride[2]
      end
      local padding = string.upper(options.padding or definition.padding or
        "VALID")
      local layer = tf.layers.max_pool2d(graph, output,
        { kernel_h, kernel_w }, { stride = { stride_h, stride_w },
          padding = padding, name = options.name })
      local function output_dim(size, kernel, stride)
        if padding == "SAME" then return math.ceil(size / stride) end
        return math.floor((size - kernel) / stride) + 1
      end
      spatial = { output_dim(spatial[1], kernel_h, stride_h),
        output_dim(spatial[2], kernel_w, stride_w), spatial[3] }
      feature_size = spatial[1] * spatial[2] * spatial[3]
      layers[i] = { output = layer.output, input_size = feature_size,
        units = spatial[3], kind = "max_pool2d",
        kernel_size = { kernel_h, kernel_w } }
      output = layer.output
    elseif definition.type == "avg_pool2d" then
      if #spatial ~= 3 then
        error("avg_pool2d layers require an NHWC input", 3)
      end
      local kernel_h, kernel_w = definition.kernel_size,
        definition.kernel_size
      if type(definition.kernel_size) == "table" then
        kernel_h, kernel_w = definition.kernel_size[1], definition.kernel_size[2]
      end
      local stride_h, stride_w = definition.stride or kernel_h,
        definition.stride or kernel_w
      if type(definition.stride) == "table" then
        stride_h, stride_w = definition.stride[1], definition.stride[2]
      end
      local padding = string.upper(options.padding or definition.padding or
        "VALID")
      local layer = tf.layers.avg_pool2d(graph, output,
        { kernel_h, kernel_w }, { stride = { stride_h, stride_w },
          padding = padding, name = options.name })
      local function output_dim(size, kernel, stride)
        if padding == "SAME" then return math.ceil(size / stride) end
        return math.floor((size - kernel) / stride) + 1
      end
      spatial = { output_dim(spatial[1], kernel_h, stride_h),
        output_dim(spatial[2], kernel_w, stride_w), spatial[3] }
      feature_size = spatial[1] * spatial[2] * spatial[3]
      layers[i] = { output = layer.output, input_size = feature_size,
        units = spatial[3], kind = "avg_pool2d",
        kernel_size = { kernel_h, kernel_w } }
      output = layer.output
    elseif definition.type == "global_avg_pool2d" then
      if #spatial ~= 3 then
        error("global_avg_pool2d layers require an NHWC input", 3)
      end
      local layer = tf.layers.global_avg_pool2d(graph, output,
        { name = options.name })
      spatial = { spatial[3] }
      feature_size = spatial[1]
      layers[i] = { output = layer.output, input_size = feature_size,
        units = spatial[1], kind = "global_avg_pool2d" }
      output = layer.output
    elseif definition.type == "dropout" then
      local layer = tf.layers.dropout(graph, output,
        definition.rate or options.rate or 0.5, {
          training = training, seed = options.seed, seed2 = options.seed2,
          name = options.name, dtype = options.dtype,
        })
      layers[i] = { output = layer.output, input_size = feature_size,
        units = feature_size, kind = "dropout" }
      output = layer.output
    elseif definition.type == "embedding" then
      local input_dim = definition.input_dim or options.input_dim
      local output_dim = definition.output_dim or options.output_dim
      if type(input_dim) ~= "number" or type(output_dim) ~= "number" then
        error("embedding layers require input_dim and output_dim", 3)
      end
      options.indices_dtype = self._input_dtype or "int32"
      local layer = tf.layers.embedding(graph, output, input_dim, output_dim,
        options)
      local total = feature_size
      if i == 1 then
        total = 1
        for _, dimension in ipairs(self._input_shape) do
          total = total * dimension
        end
      end
      layers[i] = { output = layer.output, input_size = total,
        units = total * output_dim, kind = "embedding",
        input_dim = input_dim, output_dim = output_dim }
      for _, variable in ipairs(layer.variables) do
        variables[#variables + 1] = variable
      end
      add_initializers(initializers, layer.initializers)
      output = layer.output
      feature_size = total * output_dim
      spatial = { feature_size }
    elseif definition.type == "flatten" then
      local total = feature_size
      if i == 1 then
        total = 1
        for _, dimension in ipairs(self._input_shape) do
          total = total * dimension
        end
      end
      local layer = tf.layers.flatten(graph, output, total, options)
      layers[i] = { output = layer.output, input_size = total,
        units = total, kind = "flatten" }
      output = layer.output
      feature_size = total
      spatial = { total }
    else
      options.input_size = options.input_size or feature_size
      local layer = tf.layers.dense(graph, output, definition.units, options)
      layers[i] = layer
      layers[i].kind = "dense"
      for _, variable in ipairs(layer.variables) do
        variables[#variables + 1] = variable
      end
      add_initializers(initializers, layer.initializers)
      output = layer.output
      feature_size = definition.units
      spatial = { definition.units }
    end
  end
  local units = feature_size
  local loss_spec = self._compiled.loss
  local sparse = loss_spec == "sparse_softmax_cross_entropy" or
    loss_spec == "sparse_ce" or
    loss_spec == "sparse_categorical_cross_entropy"
  local target = tf.placeholder(graph, sparse and "int32" or "float32",
    sparse and { -1 } or { -1, units }, "target")
  local sample_weight = tf.placeholder(graph, "float32", { -1 },
    "sample_weight")

  local loss
  local function weighted_mean(per_example)
    local weighted = tf.mul(graph, per_example, sample_weight,
      "sample_weighted_loss", "float32")
    return tf.reduce_mean(graph, weighted, { 0 }, "sample_weighted_mean",
      "float32")
  end
  if type(loss_spec) == "function" then
    loss = loss_spec(graph, output, target)
  elseif loss_spec == "mse" or loss_spec == "mean_squared_error" then
    loss = weighted_mean(tf.losses.mean_squared_error(graph, output, target,
      { axes = { 1 } }))
  elseif loss_spec == "softmax_cross_entropy" or loss_spec == "cce" then
    loss = weighted_mean(tf.losses.softmax_cross_entropy(graph, output, target,
      { per_example = true }))
  elseif loss_spec == "mae" or loss_spec == "mean_absolute_error" then
    loss = weighted_mean(tf.losses.mean_absolute_error(graph, output, target,
      { axes = { 1 } }))
  elseif loss_spec == "binary_cross_entropy" or loss_spec == "bce" then
    loss = weighted_mean(tf.losses.binary_cross_entropy(graph, output, target,
      { axes = { 1 } }))
  elseif sparse then
    local one_hot = tf.one_hot(graph, target, units, { name = "target_one_hot" })
    local smoothing = self._compiled.label_smoothing or 0
    if smoothing > 0 then
      if smoothing >= 1 then
        error("label_smoothing must be in [0, 1)", 3)
      end
      local smooth = tf.constant(graph, smoothing / units,
        "label_smoothing_value", "float32")
      local kept = tf.constant(graph, 1 - smoothing,
        "label_smoothing_keep", "float32")
      one_hot = tf.add(graph, tf.mul(graph, one_hot, kept,
        "label_smoothing_scale", "float32"), smooth,
        "label_smoothing_labels", "float32")
    end
    local ce = graph:new_operation("SoftmaxCrossEntropyWithLogits",
      "sparse_loss")
    ce:add_input(output)
    ce:add_input(one_hot)
    ce:set_type("T", "float32")
    local per_example = ce:finish()
    local weights = sample_weight
    local class_weight = self._compiled.class_weight
    if class_weight ~= nil then
      local class_weights = {}
      for i = 0, units - 1 do
        class_weights[#class_weights + 1] = class_weight[i] or 1
      end
      local axis = tf.constant(graph, 0, "class_axis", "int32")
      local gather = graph:new_operation("GatherV2", "class_sample_weights")
      gather:add_input(tf.constant(graph, class_weights, "class_weights",
        "float32"))
      gather:add_input(target)
      gather:add_input(axis)
      gather:set_type("Tparams", "float32")
      gather:set_type("Tindices", "int32")
      weights = tf.mul(graph, gather:finish(), sample_weight,
        "class_and_sample_weights", "float32")
    end
    local weighted = tf.mul(graph, per_example, weights,
      "weighted_sparse_loss", "float32")
    loss = tf.reduce_mean(graph, weighted, { 0 }, "sparse_mean", "float32")
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
    elseif name:match("^top_(%d+)_accuracy$") then
      local k = tonumber(name:match("^top_(%d+)_accuracy$"))
      if k < 1 or k > units then
        error("top-k metric k must be between 1 and the number of units", 3)
      end
      local top = graph:new_operation("TopKV2", "metric/topk")
      top:add_input(output)
      top:add_input(tf.constant(graph, k, "metric/topk_k", "int32"))
      top:set_type("T", "float32")
      top:set_bool("sorted", true)
      local top_output = top:finish()
      local indices = { oper = top_output.oper, index = 1,
        name = "metric/topk_indices" }
      local labels
      if sparse then
        labels = target
      else
        labels = tf.cast(graph, tf.argmax(graph, target, 1,
          "metric/topk_label", "float32", "int64"), "int64", "int32",
          "metric/topk_label_i32")
      end
      local expanded = graph:new_operation("ExpandDims",
        "metric/topk_label_axis")
      expanded:add_input(labels)
      expanded:add_input(tf.constant(graph, 1, "metric/topk_axis", "int32"))
      expanded:set_type("T", "int32")
      expanded:set_type("Tdim", "int32")
      local label_matrix = expanded:finish()
      local equal = graph:new_operation("Equal", "metric/topk_equal")
      equal:add_input(label_matrix)
      equal:add_input(indices)
      equal:set_type("T", "int32")
      local matches = equal:finish()
      local any = graph:new_operation("Any", "metric/topk_any")
      any:add_input(matches)
      any:add_input(tf.constant(graph, 1, "metric/topk_reduce_axis",
        "int32"))
      any:set_type("Tidx", "int32")
      any:set_bool("keep_dims", false)
      local hit = any:finish()
      local numeric = tf.cast(graph, hit, "bool", "float32",
        "metric/topk_float")
      metric_outputs[i] = tf.reduce_mean(graph, numeric, { 0 },
        "metric/topk_accuracy", "float32")
      metric_names[i] = name
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
  elseif optimizer_type == "centered_rmsprop" then
    local result = { tf.train.centered_rmsprop(graph, loss, variables,
      optimizer) }
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
  elseif optimizer_type == "proximal_adagrad" then
    local result = { tf.train.proximal_adagrad(graph, loss, variables,
      optimizer) }
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
  self._training = training
  self._sample_weight = sample_weight
  self._updates, self._variables, self._initializers = targets, variables,
    initializers
  self._optimizer_state = type(state) == "table" and state or nil
  self._layers_built = layers
  self._units = units
  self._sparse = sparse
  if units > 1 then
    self._class_output = tf.argmax(graph, output, 1, "prediction/class",
      "float32", "int64")
    self._proba_output = tf.nn.softmax(graph, output,
      "prediction/probabilities", "float32")
  else
    self._class_output = nil
    self._proba_output = nil
  end
  self._metric_names, self._metric_outputs = metric_names, metric_outputs
  self._batchnorm_updates = batchnorm_updates
  self._batchnorm_state = batchnorm_state
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
  local sample_weights = options.sample_weight
  if sample_weights ~= nil and #sample_weights ~= #x then
    error("sample_weight must have one value per sample", 2)
  end
  if self._session == nil then self._graph = nil end
  if options.validation_split ~= nil and options.validation_data == nil then
    local fraction = options.validation_split
    if type(fraction) ~= "number" or fraction <= 0 or fraction >= 1 then
      error("validation_split must be between 0 and 1", 2)
    end
    local cut = math.floor(#x * (1 - fraction))
    if cut < 1 or cut >= #x then
      error("validation_split produces an empty partition", 2)
    end
    local validation_x, validation_y = {}, {}
    for i = cut + 1, #x do
      validation_x[#validation_x + 1] = x[i]
      validation_y[#validation_y + 1] = y[i]
    end
    local train_x, train_y, train_weights = {}, {}, {}
    for i = 1, cut do
      train_x[#train_x + 1] = x[i]
      train_y[#train_y + 1] = y[i]
      if sample_weights ~= nil then
        train_weights[#train_weights + 1] = sample_weights[i]
      end
    end
    x, y = train_x, train_y
    if sample_weights ~= nil then sample_weights = train_weights end
    options.validation_data = { validation_x, validation_y }
  end
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
    local shuffled_x, shuffled_y, shuffled_weights = {}, {}, {}
    for i, index in ipairs(order) do
      shuffled_x[i] = x[index]
      shuffled_y[i] = y[index]
      if sample_weights ~= nil then
        shuffled_weights[i] = sample_weights[index]
      end
    end
    x, y = shuffled_x, shuffled_y
    if sample_weights ~= nil then sample_weights = shuffled_weights end
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
  self._epoch = self._epoch or 0
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
      local batch_weights = {}
      if sample_weights ~= nil then
        for i = first, last do batch_weights[#batch_weights + 1] = sample_weights[i] end
      else
        for _ = first, last do batch_weights[#batch_weights + 1] = 1 end
      end
      local feeds = { [self._input] = batch_x, [self._target] = batch_y,
        [self._sample_weight] = batch_weights }
      if self._training ~= nil then feeds[self._training] = true end
      if #(self._batchnorm_updates or {}) > 0 then
        self._session:run(feeds, {}, self._batchnorm_updates)
      end
      local values = self._session:run(feeds, fetch, self._updates)
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
    self._epoch = self._epoch + 1
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

function Sequential:predict_proba(x)
  if self._units ~= nil and self._units <= 1 then
    error("predict_proba requires more than one output unit", 2)
  end
  if self._session == nil then self._graph = nil end
  self:_build()
  if is_dataset(x) then x = x:data() end
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  if #x == 0 then error("predict_proba requires non-empty input", 2) end
  local feeds = { [self._input] = x }
  if self._training ~= nil then feeds[self._training] = false end
  return self._session:run(feeds, { self._proba_output })[1]
end

function Sequential:predict_classes(x)
  if self._units ~= nil and self._units <= 1 then
    error("predict_classes requires more than one output unit", 2)
  end
  if self._session == nil then self._graph = nil end
  self:_build()
  if is_dataset(x) then x = x:data() end
  x = normalize_rows(x, self._input_shape[#self._input_shape])
  if #x == 0 then error("predict_classes requires non-empty input", 2) end
  local feeds = { [self._input] = x }
  if self._training ~= nil then feeds[self._training] = false end
  return self._session:run(feeds, { self._class_output })[1]
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
  local feeds = { [self._input] = x }
  if self._training ~= nil then feeds[self._training] = false end
  return self._session:run(feeds, { self._output })[1]
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
  local sample_weights = options.sample_weight
  if sample_weights ~= nil and #sample_weights ~= #x then
    error("sample_weight must have one value per sample", 2)
  end
  local batch_size = options.batch_size or #x
  local total, batches = 0, 0
  local metric_sums = {}
  for first = 1, #x, batch_size do
    local last = math.min(first + batch_size - 1, #x)
    local fetch = { self._loss }
    for _, output in ipairs(self._metric_outputs or {}) do
      fetch[#fetch + 1] = output
    end
    local batch_weights = {}
    if sample_weights ~= nil then
      for i = first, last do
        batch_weights[#batch_weights + 1] = sample_weights[i]
      end
    else
      for _ = first, last do batch_weights[#batch_weights + 1] = 1 end
    end
    local feeds = { [self._input] = batch_slice(x, first, last),
      [self._target] = batch_slice(y, first, last),
      [self._sample_weight] = batch_weights }
    if self._training ~= nil then feeds[self._training] = false end
    local values = self._session:run(feeds, fetch)
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

local function checkpoint_entries(variables)
  local entries = {}
  for _, variable in ipairs(variables) do
    entries[#entries + 1] = { name = variable.name, variable = variable }
  end
  return entries
end

function Sequential:_training_entries()
  local entries = checkpoint_entries(self._variables)
  local state = self._optimizer_state
  if state ~= nil and type(state.variables) == "table" then
    local optimizer_entries = checkpoint_entries(state.variables)
    for _, entry in ipairs(optimizer_entries) do
      entries[#entries + 1] = entry
    end
  end
  local batchnorm_entries = checkpoint_entries(self._batchnorm_state or {})
  for _, entry in ipairs(batchnorm_entries) do
    entries[#entries + 1] = entry
  end
  return entries
end

function Sequential:save_training(prefix)
  if self._session == nil then error("model has no session to save", 2) end
  local tf = require("tensorflow")
  self._session:run({}, {}, { tf.save_v2(self._graph, prefix,
    self:_training_entries()) })
  local meta, err = io.open(prefix .. ".state", "wb")
  if meta == nil then error("cannot write training state: " .. tostring(err), 2) end
  meta:write(string.format("epoch=%d\n", self._epoch or 0))
  meta:close()
  return prefix
end

function Sequential:load_training(prefix)
  if self._session == nil then self._graph = nil end
  self:_build()
  local tf = require("tensorflow")
  self._session:run({}, {}, tf.restore_v2(self._graph, prefix,
    self:_training_entries()))
  local meta = io.open(prefix .. ".state", "rb")
  if meta ~= nil then
    local contents = meta:read("*a")
    meta:close()
    local epoch = contents:match("epoch=(%d+)")
    if epoch ~= nil then self._epoch = tonumber(epoch) end
  end
  return self
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

function Sequential:get_weights()
  if self._session == nil then self._graph = nil end
  self:_build()
  local fetch = {}
  for i, variable in ipairs(self._variables) do fetch[i] = variable.read end
  return self._session:run({}, fetch)
end

function Sequential:set_weights(weights)
  if self._session == nil then self._graph = nil end
  self:_build()
  if type(weights) ~= "table" or #weights ~= #self._variables then
    error("set_weights expects one tensor per variable", 2)
  end
  local tf = require("tensorflow")
  self._set_counter = (self._set_counter or 0) + 1
  local targets = {}
  for i, variable in ipairs(self._variables) do
    local constant = tf.constant(self._graph, weights[i],
      "set_weight_value_" .. self._set_counter .. "_" .. i, variable.dtype)
    local desc = self._graph:new_operation("Assign",
      "set_weight_" .. self._set_counter .. "_" .. i)
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

function Sequential:count_params()
  if self._session == nil then self._graph = nil end
  self:_build()
  local total = 0
  for _, variable in ipairs(self._variables) do
    local count = 1
    for _, dimension in ipairs(variable.shape) do count = count * dimension end
    total = total + count
  end
  return total
end

function Sequential:summary()
  self:_build()
  local rows = { "Sequential" }
  for i, layer in ipairs(self._layers_built) do
    if layer.kind == "flatten" then
      rows[#rows + 1] = string.format("  %d: Flatten(%d)", i,
        layer.input_size)
    elseif layer.kind == "conv1d" then
      rows[#rows + 1] = string.format("  %d: Conv1D(%d filters, %d)", i,
        layer.filters, layer.kernel_size)
    elseif layer.kind == "conv2d" then
      rows[#rows + 1] = string.format("  %d: Conv2D(%d filters, %dx%d)", i,
        layer.filters, layer.kernel_size[1], layer.kernel_size[2])
    elseif layer.kind == "max_pool2d" then
      rows[#rows + 1] = string.format("  %d: MaxPool2D(%dx%d)", i,
        layer.kernel_size[1], layer.kernel_size[2])
    elseif layer.kind == "avg_pool2d" then
      rows[#rows + 1] = string.format("  %d: AvgPool2D(%dx%d)", i,
        layer.kernel_size[1], layer.kernel_size[2])
    elseif layer.kind == "global_avg_pool2d" then
      rows[#rows + 1] = string.format("  %d: GlobalAvgPool2D(%d)", i,
        layer.units)
    elseif layer.kind == "batchnorm" then
      rows[#rows + 1] = string.format("  %d: BatchNormalization", i)
    elseif layer.kind == "dropout" then
      rows[#rows + 1] = string.format("  %d: Dropout", i)
    elseif layer.kind == "embedding" then
      rows[#rows + 1] = string.format("  %d: Embedding(%d -> %d)", i,
        layer.input_dim, layer.output_dim)
    else
      rows[#rows + 1] = string.format("  %d: Dense(%d -> %d)", i,
        layer.input_size, layer.units)
    end
  end
  rows[#rows + 1] = string.format("Total params: %d", self:count_params())
  return table.concat(rows, "\n")
end

return Sequential
