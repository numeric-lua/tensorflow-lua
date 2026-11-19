--
-- Small compositional layer builders on top of graph variables.
--
local Layers = {}

local function random_uniform(rows, columns, bound)
  local values = {}
  for r = 1, rows do
    values[r] = {}
    for c = 1, columns do
      values[r][c] = (math.random() * 2 - 1) * bound
    end
  end
  return values
end

local function gather_dimension(graph, shape, index, name)
  local tf = require("tensorflow")
  local indices = tf.constant(graph, { index }, name .. "/index", "int32")
  local axis = tf.constant(graph, 0, name .. "/axis", "int32")
  local desc = graph:new_operation("GatherV2", name)
  desc:add_input(shape)
  desc:add_input(indices)
  desc:add_input(axis)
  desc:set_type("Tparams", "int32")
  desc:set_type("Tindices", "int32")
  desc:set_int("batch_dims", 0)
  return desc:finish()
end

local function dimension_vector(graph, value, name)
  if type(value) == "number" then
    return require("tensorflow").constant(graph, { value }, name, "int32")
  end
  return value
end

function Layers.dense(graph, input, units, options)
  options = options or {}
  local input_size = options.input_size or options.in_features
  if type(input_size) ~= "number" or input_size < 1 then
    error("dense requires options.input_size", 3)
  end
  if type(units) ~= "number" or units < 1 then
    error("dense requires a positive unit count", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "dense"
  local tf = require("tensorflow")
  local bound = math.sqrt(6 / (input_size + units))
  local weight = tf.variable(graph,
    random_uniform(input_size, units, bound), prefix .. "/weight",
    { dtype = dtype })
  local variables = { weight }
  local initializers = { weight.initializer }
  local bias = nil
  local output = tf.matmul(graph, input, weight.read, prefix .. "/matmul",
    dtype)
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, units do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  local activation = options.activation
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. activation, dtype)
  elseif type(activation) == "function" then
    output = activation(graph, output, prefix .. "/activation", dtype)
  elseif activation ~= nil then
    error("activation must be a string, function, or nil", 3)
  end
  return {
    output = output,
    weight = weight,
    bias = bias,
    variables = variables,
    initializers = initializers,
    input_size = input_size,
    units = units,
    dtype = dtype,
  }
end

function Layers.batch_normalization(graph, input, channels, options)
  options = options or {}
  if type(channels) ~= "number" or channels < 1 then
    error("batch_normalization requires a positive channel count", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "batch_normalization"
  local axes = options.axes or { 0 }
  local epsilon = options.epsilon or 1e-5
  local momentum = options.momentum or 0.99
  local tf = require("tensorflow")
  local ones, zeros = {}, {}
  for i = 1, channels do ones[i] = 1; zeros[i] = 0 end
  local gamma = tf.variable(graph, ones, prefix .. "/gamma", { dtype = dtype })
  local beta = tf.variable(graph, zeros, prefix .. "/beta", { dtype = dtype })
  local moving_mean = tf.variable(graph, zeros, prefix .. "/moving_mean",
    { dtype = dtype })
  local moving_variance = tf.variable(graph, ones,
    prefix .. "/moving_variance", { dtype = dtype })
  local initializers = { gamma.initializer, beta.initializer,
    moving_mean.initializer, moving_variance.initializer }

  local batch_mean = tf.reduce_mean(graph, input, axes,
    prefix .. "/batch_mean", dtype, true)
  local centered = tf.sub(graph, input, batch_mean, prefix .. "/centered", dtype)
  local squared = tf.square(graph, centered, prefix .. "/squared", dtype)
  local batch_variance = tf.reduce_mean(graph, squared, axes,
    prefix .. "/batch_variance", dtype, true)

  local training = options.training
  local gate
  if type(training) == "table" and training.oper ~= nil then
    gate = tf.cast(graph, training, "bool", dtype, prefix .. "/training")
  else
    gate = tf.constant(graph, training == false and 0 or 1,
      prefix .. "/training", dtype)
  end
  local mean_delta = tf.sub(graph, batch_mean, moving_mean.read,
    prefix .. "/mean_delta", dtype)
  local mean = tf.add(graph, moving_mean.read,
    tf.mul(graph, gate, mean_delta, prefix .. "/mean_gate", dtype),
    prefix .. "/mean", dtype)
  local variance_delta = tf.sub(graph, batch_variance,
    moving_variance.read, prefix .. "/variance_delta", dtype)
  local variance = tf.add(graph, moving_variance.read,
    tf.mul(graph, gate, variance_delta, prefix .. "/variance_gate", dtype),
    prefix .. "/variance", dtype)
  local epsilon_op = tf.constant(graph, epsilon, prefix .. "/epsilon", dtype)
  local variance_floor = tf.clip_by_value(graph, variance, epsilon, 1e30,
    prefix .. "/variance_floor", dtype)
  local denominator = tf.sqrt(graph, tf.add(graph, variance_floor, epsilon_op,
    prefix .. "/variance_epsilon", dtype), prefix .. "/denominator", dtype)
  local normalized = tf.div(graph, tf.sub(graph, input, mean,
    prefix .. "/normalized_delta", dtype), denominator,
    prefix .. "/normalized", dtype)
  local scaled = tf.mul(graph, normalized, gamma.read, prefix .. "/scaled",
    dtype)
  local output = tf.add(graph, scaled, beta.read, prefix .. "/shifted", dtype)

  local decay = tf.constant(graph, momentum, prefix .. "/decay", dtype)
  local one_minus_decay = tf.constant(graph, 1 - momentum,
    prefix .. "/one_minus_decay", dtype)
  local mean_flat = tf.reshape(graph, batch_mean, { channels },
    prefix .. "/mean_flat", dtype)
  local variance_flat = tf.reshape(graph, batch_variance, { channels },
    prefix .. "/variance_flat", dtype)
  local new_mean = tf.add(graph,
    tf.mul(graph, moving_mean.read, decay, prefix .. "/mean_decay", dtype),
    tf.mul(graph, mean_flat, one_minus_decay,
      prefix .. "/mean_update", dtype), prefix .. "/new_mean", dtype)
  local new_variance = tf.add(graph,
    tf.mul(graph, moving_variance.read, decay,
      prefix .. "/variance_decay", dtype),
    tf.mul(graph, variance_flat, one_minus_decay,
      prefix .. "/variance_update", dtype), prefix .. "/new_variance", dtype)
  local gated_mean = tf.add(graph, moving_mean.read,
    tf.mul(graph, gate, tf.sub(graph, new_mean, moving_mean.read,
      prefix .. "/new_mean_delta", dtype), prefix .. "/mean_gate_update",
      dtype), prefix .. "/gated_mean", dtype)
  local gated_variance = tf.add(graph, moving_variance.read,
    tf.mul(graph, gate, tf.sub(graph, new_variance, moving_variance.read,
      prefix .. "/new_variance_delta", dtype),
      prefix .. "/variance_gate_update", dtype),
    prefix .. "/gated_variance", dtype)
  local update_mean = graph:new_operation("Assign", prefix .. "/update_mean")
  update_mean:add_input(moving_mean.ref)
  update_mean:add_input(gated_mean)
  update_mean:set_type("T", dtype)
  update_mean:set_bool("use_locking", true)
  local mean_assign = update_mean:finish()
  local update_variance = graph:new_operation("Assign",
    prefix .. "/update_variance")
  update_variance:add_input(moving_variance.ref)
  update_variance:add_input(gated_variance)
  update_variance:set_type("T", dtype)
  update_variance:set_bool("use_locking", true)
  local variance_assign = update_variance:finish()

  local final = graph:new_operation("Identity", prefix .. "/output")
  final:add_input(output)
  final:set_type("T", dtype)
  return {
    output = final:finish(),
    updates = { mean_assign, variance_assign },
    gamma = gamma,
    beta = beta,
    moving_mean = moving_mean,
    moving_variance = moving_variance,
    variables = { gamma, beta },
    initializers = initializers,
    channels = channels,
    axes = axes,
    dtype = dtype,
  }
end

function Layers.conv2d(graph, input, filters, kernel_size, options)
  options = options or {}
  if type(filters) ~= "number" or filters < 1 then
    error("conv2d requires positive filters", 3)
  end
  local in_channels = options.in_channels
  if type(in_channels) ~= "number" or in_channels < 1 then
    error("conv2d requires options.in_channels", 3)
  end
  local kernel_h, kernel_w = kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_h, kernel_w = kernel_size[1], kernel_size[2]
  end
  local stride_h = options.stride or 1
  local stride_w = options.stride or 1
  if type(options.stride) == "table" then
    stride_h, stride_w = options.stride[1], options.stride[2]
  end
  local dilation_h = options.dilation or 1
  local dilation_w = options.dilation or 1
  if type(options.dilation) == "table" then
    dilation_h, dilation_w = options.dilation[1], options.dilation[2]
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "conv2d"
  local tf = require("tensorflow")
  local bound = math.sqrt(6 / (kernel_h * kernel_w * in_channels + filters))
  local values = {}
  for i = 1, kernel_h do
    values[i] = {}
    for j = 1, kernel_w do
      values[i][j] = {}
      for c = 1, in_channels do
        values[i][j][c] = {}
        for f = 1, filters do
          values[i][j][c][f] = (math.random() * 2 - 1) * bound
        end
      end
    end
  end
  local weight = tf.variable(graph, values, prefix .. "/weight",
    { dtype = dtype })
  local desc = graph:new_operation("Conv2D", prefix .. "/conv")
  desc:add_input(input)
  desc:add_input(weight.read)
  desc:set_type("T", dtype)
  desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
  desc:set_string("padding", options.padding or "SAME")
  desc:set_string("data_format", "NHWC")
  desc:set_int_list("dilations", { 1, dilation_h, dilation_w, 1 })
  desc:set_bool("use_cudnn_on_gpu", false)
  local output = desc:finish()
  local variables, initializers = { weight }, { weight.initializer }
  local bias = nil
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, filters do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  local activation = options.activation
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. activation, dtype)
  elseif type(activation) == "function" then
    output = activation(graph, output, prefix .. "/activation", dtype)
  end
  return {
    output = output,
    weight = weight,
    bias = bias,
    variables = variables,
    initializers = initializers,
    filters = filters,
    kernel_size = { kernel_h, kernel_w },
    dtype = dtype,
  }
end

function Layers.conv2d_transpose(graph, input, filters, kernel_size, options)
  options = options or {}
  if type(filters) ~= "number" or filters < 1 then
    error("conv2d_transpose requires positive filters", 3)
  end
  local in_channels = options.in_channels
  if type(in_channels) ~= "number" or in_channels < 1 then
    error("conv2d_transpose requires options.in_channels", 3)
  end
  local kernel_h, kernel_w = kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_h, kernel_w = kernel_size[1], kernel_size[2]
  end
  if type(kernel_h) ~= "number" or type(kernel_w) ~= "number" or
      kernel_h < 1 or kernel_w < 1 then
    error("conv2d_transpose requires a positive kernel size", 3)
  end
  local stride_h, stride_w = options.stride or 1, options.stride or 1
  if type(options.stride) == "table" then
    stride_h, stride_w = options.stride[1], options.stride[2]
  end
  if type(stride_h) ~= "number" or type(stride_w) ~= "number" or
      stride_h < 1 or stride_w < 1 then
    error("conv2d_transpose requires a positive stride", 3)
  end
  local dilation_h, dilation_w = options.dilation or 1, options.dilation or 1
  if type(options.dilation) == "table" then
    dilation_h, dilation_w = options.dilation[1], options.dilation[2]
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "conv2d_transpose"
  local padding = string.upper(options.padding or "SAME")
  local data_format = string.upper(options.data_format or "NHWC")
  if padding ~= "SAME" and padding ~= "VALID" then
    error("conv2d_transpose padding must be SAME or VALID", 3)
  end
  if data_format ~= "NHWC" and data_format ~= "NCHW" then
    error("conv2d_transpose data_format must be NHWC or NCHW", 3)
  end
  local tf = require("tensorflow")

  local output_shape = options.output_shape
  local output_batch, output_h, output_w, output_c
  if output_shape ~= nil then
    if type(output_shape) ~= "table" then
      error("conv2d_transpose output_shape must be a table", 3)
    end
    if #output_shape == 4 then
      if data_format == "NCHW" then
        output_batch, output_c = output_shape[1], output_shape[2]
        output_h, output_w = output_shape[3], output_shape[4]
      else
        output_batch, output_h = output_shape[1], output_shape[2]
        output_w, output_c = output_shape[3], output_shape[4]
      end
    elseif #output_shape == 3 then
      output_h, output_w, output_c = output_shape[1], output_shape[2],
        output_shape[3]
    else
      error("conv2d_transpose output_shape must have 3 or 4 dimensions", 3)
    end
  end

  local input_shape = options.input_shape
  local input_h, input_w
  if input_shape ~= nil then
    if type(input_shape) ~= "table" or
        (#input_shape ~= 3 and #input_shape ~= 4) then
      error("conv2d_transpose input_shape must have 3 or 4 dimensions", 3)
    end
    if #input_shape == 4 then
      if data_format == "NCHW" then
        input_h, input_w = input_shape[3], input_shape[4]
      else
        input_h, input_w = input_shape[2], input_shape[3]
      end
    else
      input_h, input_w = input_shape[1], input_shape[2]
    end
  end

  local shape_op
  local function dynamic_input_shape()
    shape_op = shape_op or tf.shape(graph, input, prefix .. "/input_shape")
    return shape_op
  end
  local effective_h = (kernel_h - 1) * dilation_h + 1
  local effective_w = (kernel_w - 1) * dilation_w + 1
  local function output_dimension(size, stride, effective, name)
    if padding == "SAME" then
      if type(size) == "number" then return size * stride end
      return tf.mul(graph, size,
        tf.constant(graph, stride, name .. "/stride", "int32"),
        name, "int32")
    end
    if type(size) == "number" then
      return (size - 1) * stride + effective
    end
    local one = tf.constant(graph, 1, name .. "/one", "int32")
    local offset = tf.constant(graph, effective, name .. "/effective", "int32")
    local scaled = tf.mul(graph, tf.sub(graph, size, one, name .. "/minus_one",
      "int32"), tf.constant(graph, stride, name .. "/stride", "int32"),
      name .. "/scaled", "int32")
    return tf.add(graph, scaled, offset, name, "int32")
  end
  if output_h == nil then
    local height_index = data_format == "NCHW" and 2 or 1
    local width_index = data_format == "NCHW" and 3 or 2
    local height, width = input_h, input_w
    if height == nil then
      height = gather_dimension(graph, dynamic_input_shape(), height_index,
        prefix .. "/input_height")
    end
    if width == nil then
      width = gather_dimension(graph, dynamic_input_shape(), width_index,
        prefix .. "/input_width")
    end
    output_h = output_dimension(height, stride_h, effective_h,
      prefix .. "/output_height")
    output_w = output_dimension(width, stride_w, effective_w,
      prefix .. "/output_width")
  end
  output_c = output_c or filters
  if type(output_c) ~= "number" or output_c ~= filters then
    error("conv2d_transpose output_shape channels must equal filters", 3)
  end

  local batch
  if type(output_batch) == "number" and output_batch > 0 then
    batch = dimension_vector(graph, output_batch, prefix .. "/batch_size")
  else
    batch = gather_dimension(graph, dynamic_input_shape(), 0,
      prefix .. "/batch_size")
  end
  local height = dimension_vector(graph, output_h, prefix .. "/height")
  local width = dimension_vector(graph, output_w, prefix .. "/width")
  local channels = dimension_vector(graph, output_c, prefix .. "/channels")
  local dims
  if data_format == "NCHW" then
    dims = { batch, channels, height, width }
  else
    dims = { batch, height, width, channels }
  end
  local output_sizes = tf.concat(graph, dims, 0, prefix .. "/output_shape",
    "int32")

  local bound = math.sqrt(6 / (kernel_h * kernel_w * in_channels + filters))
  local values = {}
  for i = 1, kernel_h do
    values[i] = {}
    for j = 1, kernel_w do
      values[i][j] = {}
      for f = 1, filters do
        values[i][j][f] = {}
        for c = 1, in_channels do
          values[i][j][f][c] = (math.random() * 2 - 1) * bound
        end
      end
    end
  end
  local weight = tf.variable(graph, values, prefix .. "/weight",
    { dtype = dtype })
  local desc = graph:new_operation("Conv2DBackpropInput",
    prefix .. "/conv_transpose")
  desc:add_input(output_sizes)
  desc:add_input(weight.read)
  desc:add_input(input)
  desc:set_type("T", dtype)
  if data_format == "NCHW" then
    desc:set_int_list("strides", { 1, 1, stride_h, stride_w })
    desc:set_int_list("dilations", { 1, 1, dilation_h, dilation_w })
  else
    desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
    desc:set_int_list("dilations", { 1, dilation_h, dilation_w, 1 })
  end
  desc:set_string("padding", padding)
  desc:set_string("data_format", data_format)
  desc:set_bool("use_cudnn_on_gpu", false)
  local output = desc:finish()
  local variables, initializers = { weight }, { weight.initializer }
  local bias = nil
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, filters do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  local activation = options.activation
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. activation, dtype)
  elseif type(activation) == "function" then
    output = activation(graph, output, prefix .. "/activation", dtype)
  end
  return {
    output = output,
    weight = weight,
    bias = bias,
    variables = variables,
    initializers = initializers,
    filters = filters,
    in_channels = in_channels,
    kernel_size = { kernel_h, kernel_w },
    stride = { stride_h, stride_w },
    padding = padding,
    output_shape = { output_h, output_w, output_c },
    data_format = data_format,
    dtype = dtype,
  }
end

function Layers.conv1d(graph, input, filters, kernel_size, options)
  options = options or {}
  if type(filters) ~= "number" or filters < 1 then
    error("conv1d requires positive filters", 3)
  end
  local in_channels = options.in_channels
  if type(in_channels) ~= "number" or in_channels < 1 then
    error("conv1d requires options.in_channels", 3)
  end
  local kernel = kernel_size
  local stride = options.stride or 1
  local dilation = options.dilation or 1
  local dtype = options.dtype or "float32"
  local prefix = options.name or "conv1d"
  local tf = require("tensorflow")
  local bound = math.sqrt(6 / (kernel * in_channels + filters))
  local values = { {} }
  for k = 1, kernel do
    values[1][k] = {}
    for c = 1, in_channels do
      values[1][k][c] = {}
      for f = 1, filters do
        values[1][k][c][f] = (math.random() * 2 - 1) * bound
      end
    end
  end
  local weight = tf.variable(graph, values, prefix .. "/weight",
    { dtype = dtype })
  local second_axis = tf.constant(graph, 1, prefix .. "/axis", "int32")
  local expand = graph:new_operation("ExpandDims", prefix .. "/expand")
  expand:add_input(input)
  expand:add_input(second_axis)
  expand:set_type("T", dtype)
  expand:set_type("Tdim", "int32")
  local lifted = expand:finish()
  local conv = graph:new_operation("Conv2D", prefix .. "/conv")
  conv:add_input(lifted)
  conv:add_input(weight.read)
  conv:set_type("T", dtype)
  conv:set_int_list("strides", { 1, 1, stride, 1 })
  conv:set_string("padding", options.padding or "SAME")
  conv:set_string("data_format", "NHWC")
  conv:set_int_list("dilations", { 1, 1, dilation, 1 })
  conv:set_bool("use_cudnn_on_gpu", false)
  local convolved = conv:finish()
  local squeeze = graph:new_operation("Squeeze", prefix .. "/squeeze")
  squeeze:add_input(convolved)
  squeeze:set_type("T", dtype)
  squeeze:set_int_list("squeeze_dims", { 1 })
  local output = squeeze:finish()
  local variables, initializers = { weight }, { weight.initializer }
  local bias = nil
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, filters do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  local activation = options.activation
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. activation, dtype)
  elseif type(activation) == "function" then
    output = activation(graph, output, prefix .. "/activation", dtype)
  end
  return {
    output = output,
    weight = weight,
    bias = bias,
    variables = variables,
    initializers = initializers,
    filters = filters,
    kernel_size = kernel,
    dtype = dtype,
  }
end

function Layers.conv3d(graph, input, filters, kernel_size, options)
  options = options or {}
  if type(filters) ~= "number" or filters < 1 then
    error("conv3d requires positive filters", 3)
  end
  local shape = options.input_shape
  if type(shape) ~= "table" or #shape ~= 4 then
    error("conv3d requires options.input_shape = { d, h, w, c }", 3)
  end
  local in_channels = options.in_channels or shape[4]
  if type(in_channels) ~= "number" or in_channels ~= shape[4] then
    error("conv3d input_shape channels must match options.in_channels", 3)
  end
  local kernel_d, kernel_h, kernel_w = kernel_size, kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_d, kernel_h, kernel_w = kernel_size[1], kernel_size[2],
      kernel_size[3]
  end
  local stride_d, stride_h, stride_w = options.stride or 1,
    options.stride or 1, options.stride or 1
  if type(options.stride) == "table" then
    stride_d, stride_h, stride_w = options.stride[1], options.stride[2],
      options.stride[3]
  end
  local dilation_d, dilation_h, dilation_w = options.dilation or 1,
    options.dilation or 1, options.dilation or 1
  if type(options.dilation) == "table" then
    dilation_d, dilation_h, dilation_w = options.dilation[1],
      options.dilation[2], options.dilation[3]
  end
  local padding = string.upper(options.padding or "SAME")
  if padding ~= "SAME" and padding ~= "VALID" then
    error("conv3d padding must be SAME or VALID", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "conv3d"
  local tf = require("tensorflow")

  local effective_d = (kernel_d - 1) * dilation_d + 1
  local effective_h = (kernel_h - 1) * dilation_h + 1
  local effective_w = (kernel_w - 1) * dilation_w + 1
  local function output_dim(size, effective, stride, same)
    if same then return math.ceil(size / stride) end
    return math.floor((size - effective) / stride) + 1
  end
  local out_d = output_dim(shape[1], effective_d, stride_d, padding == "SAME")
  local out_h = output_dim(shape[2], effective_h, stride_h, padding == "SAME")
  local out_w = output_dim(shape[3], effective_w, stride_w, padding == "SAME")
  if out_d < 1 or out_h < 1 or out_w < 1 then
    error("conv3d kernel is larger than the input volume", 3)
  end
  local padded_d = (out_d - 1) * stride_d + effective_d
  local padded_h = (out_h - 1) * stride_h + effective_h
  local padded_w = (out_w - 1) * stride_w + effective_w
  local pad_d = padded_d - shape[1]
  local pad_h = padded_h - shape[2]
  local pad_w = padded_w - shape[3]
  local pads = {
    { 0, 0 },
    { math.floor(pad_d / 2), pad_d - math.floor(pad_d / 2) },
    { math.floor(pad_h / 2), pad_h - math.floor(pad_h / 2) },
    { math.floor(pad_w / 2), pad_w - math.floor(pad_w / 2) },
    { 0, 0 },
  }
  local padded = input
  if pad_d + pad_h + pad_w > 0 then
    local pad_desc = graph:new_operation("Pad", prefix .. "/pad")
    pad_desc:add_input(input)
    pad_desc:add_input(tf.constant(graph, pads, prefix .. "/paddings",
      "int32"))
    pad_desc:set_type("T", dtype)
    pad_desc:set_type("Tpaddings", "int32")
    padded = pad_desc:finish()
  end

  local bound = math.sqrt(6 / (kernel_d * kernel_h * kernel_w * in_channels +
    filters))
  local values = {}
  for d = 1, kernel_d do
    values[d] = {}
    for y = 1, kernel_h do
      values[d][y] = {}
      for x = 1, kernel_w do
        values[d][y][x] = {}
        for c = 1, in_channels do
          values[d][y][x][c] = {}
          for f = 1, filters do
            values[d][y][x][c][f] = (math.random() * 2 - 1) * bound
          end
        end
      end
    end
  end
  local weight = tf.variable(graph, values, prefix .. "/weight",
    { dtype = dtype })
  local depth = padded_d - (out_d - 1) * stride_d - effective_d
  local function depth_slice(kernel_index)
    local begin = tf.constant(graph, { 0, (kernel_index - 1) * dilation_d,
      0, 0, 0 }, prefix .. "/depth_begin_" .. kernel_index, "int32")
    local finish = tf.constant(graph, { 0,
      (kernel_index - 1) * dilation_d + (out_d - 1) * stride_d + 1,
      padded_h, padded_w, in_channels },
      prefix .. "/depth_end_" .. kernel_index, "int32")
    local strides = tf.constant(graph, { 1, stride_d, 1, 1, 1 },
      prefix .. "/depth_strides_" .. kernel_index, "int32")
    local desc = graph:new_operation("StridedSlice",
      prefix .. "/depth_" .. kernel_index)
    desc:add_input(padded)
    desc:add_input(begin)
    desc:add_input(finish)
    desc:add_input(strides)
    desc:set_type("T", dtype)
    desc:set_type("Index", "int32")
    desc:set_int("begin_mask", 0)
    desc:set_int("end_mask", 1)
    desc:set_int("ellipsis_mask", 0)
    desc:set_int("new_axis_mask", 0)
    desc:set_int("shrink_axis_mask", 0)
    return desc:finish()
  end
  local function kernel_slice(kernel_index)
    local begin = tf.constant(graph, { kernel_index - 1, 0, 0, 0, 0 },
      prefix .. "/kernel_begin_" .. kernel_index, "int32")
    local size = tf.constant(graph, { 1, kernel_h, kernel_w, in_channels,
      filters }, prefix .. "/kernel_size_" .. kernel_index, "int32")
    local desc = graph:new_operation("Slice",
      prefix .. "/kernel_" .. kernel_index)
    desc:add_input(weight.read)
    desc:add_input(begin)
    desc:add_input(size)
    desc:set_type("T", dtype)
    desc:set_type("Index", "int32")
    return desc:finish()
  end
  local total
  for kernel_index = 1, kernel_d do
    local slab = depth_slice(kernel_index)
    local flat = tf.reshape(graph, slab, { -1, padded_h, padded_w,
      in_channels }, prefix .. "/flat_" .. kernel_index, dtype)
    local kernel2 = tf.reshape(graph, kernel_slice(kernel_index),
      { kernel_h, kernel_w, in_channels, filters },
      prefix .. "/kernel2_" .. kernel_index, dtype)
    local desc = graph:new_operation("Conv2D",
      prefix .. "/conv2d_" .. kernel_index)
    desc:add_input(flat)
    desc:add_input(kernel2)
    desc:set_type("T", dtype)
    desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
    desc:set_string("padding", "VALID")
    desc:set_string("data_format", "NHWC")
    desc:set_int_list("dilations", { 1, dilation_h, dilation_w, 1 })
    desc:set_bool("use_cudnn_on_gpu", false)
    local conv = desc:finish()
    local volume = tf.reshape(graph, conv, { -1, out_d, out_h, out_w,
      filters }, prefix .. "/volume_" .. kernel_index, dtype)
    total = total == nil and volume or tf.add(graph, total, volume,
      prefix .. "/sum_" .. kernel_index, dtype)
  end
  local output = total
  local variables, initializers = { weight }, { weight.initializer }
  local bias = nil
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, filters do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  if type(options.activation) == "string" then
    local fn = tf.nn[options.activation]
    if fn == nil then error("unknown activation '" .. options.activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. options.activation, dtype)
  elseif type(options.activation) == "function" then
    output = options.activation(graph, output, prefix .. "/activation", dtype)
  end
  return {
    output = output, weight = weight, bias = bias, variables = variables,
    initializers = initializers, filters = filters,
    kernel_size = { kernel_d, kernel_h, kernel_w }, dtype = dtype,
    output_shape = { out_d, out_h, out_w, filters },
  }
end

function Layers.max_pool2d(graph, input, kernel_size, options)
  options = options or {}
  local kernel_h, kernel_w = kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_h, kernel_w = kernel_size[1], kernel_size[2]
  end
  local stride_h = options.stride or kernel_h
  local stride_w = options.stride or kernel_w
  if type(options.stride) == "table" then
    stride_h, stride_w = options.stride[1], options.stride[2]
  end
  local desc = graph:new_operation("MaxPool", options.name or "max_pool2d")
  desc:add_input(input)
  desc:set_int_list("ksize", { 1, kernel_h, kernel_w, 1 })
  desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
  desc:set_string("padding", options.padding or "VALID")
  desc:set_string("data_format", "NHWC")
  return { output = desc:finish(), variables = {}, initializers = {} }
end

function Layers.avg_pool2d(graph, input, kernel_size, options)
  options = options or {}
  local kernel_h, kernel_w = kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_h, kernel_w = kernel_size[1], kernel_size[2]
  end
  local stride_h = options.stride or kernel_h
  local stride_w = options.stride or kernel_w
  if type(options.stride) == "table" then
    stride_h, stride_w = options.stride[1], options.stride[2]
  end
  local desc = graph:new_operation("AvgPool", options.name or "avg_pool2d")
  desc:add_input(input)
  desc:set_int_list("ksize", { 1, kernel_h, kernel_w, 1 })
  desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
  desc:set_string("padding", options.padding or "VALID")
  desc:set_string("data_format", "NHWC")
  return { output = desc:finish(), variables = {}, initializers = {} }
end

function Layers.global_avg_pool2d(graph, input, options)
  options = options or {}
  local tf = require("tensorflow")
  local prefix = options.name or "global_avg_pool2d"
  local pooled = tf.reduce_mean(graph, input, { 1, 2 }, prefix .. "/mean",
    "float32", true)
  local desc = graph:new_operation("Squeeze", prefix .. "/squeeze")
  desc:add_input(pooled)
  desc:set_type("T", options.dtype or "float32")
  desc:set_int_list("squeeze_dims", { 1, 2 })
  return { output = desc:finish(), variables = {}, initializers = {} }
end

function Layers.global_max_pool2d(graph, input, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "global_max_pool2d"
  local tf = require("tensorflow")
  local desc = graph:new_operation("Max", prefix .. "/max")
  desc:add_input(input)
  desc:add_input(tf.constant(graph, { 1, 2 }, prefix .. "/axes", "int32"))
  desc:set_type("T", dtype)
  desc:set_type("Tidx", "int32")
  desc:set_bool("keep_dims", true)
  local pooled = desc:finish()
  local squeeze = graph:new_operation("Squeeze", prefix .. "/squeeze")
  squeeze:add_input(pooled)
  squeeze:set_type("T", dtype)
  squeeze:set_int_list("squeeze_dims", { 1, 2 })
  return { output = squeeze:finish(), variables = {}, initializers = {} }
end

function Layers.max_pool1d(graph, input, kernel_size, options)
  options = options or {}
  local kernel = kernel_size
  if type(kernel) ~= "number" or kernel < 1 then
    error("max_pool1d requires a positive kernel size", 3)
  end
  local stride = options.stride or kernel
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "max_pool1d"
  local expand = graph:new_operation("ExpandDims", prefix .. "/expand")
  expand:add_input(input)
  expand:add_input(tf.constant(graph, 1, prefix .. "/axis", "int32"))
  expand:set_type("T", dtype)
  expand:set_type("Tdim", "int32")
  local lifted = expand:finish()
  local pool = graph:new_operation("MaxPool", prefix .. "/pool")
  pool:add_input(lifted)
  pool:set_int_list("ksize", { 1, 1, kernel, 1 })
  pool:set_int_list("strides", { 1, 1, stride, 1 })
  pool:set_string("padding", options.padding or "VALID")
  pool:set_string("data_format", "NHWC")
  local pooled = pool:finish()
  local squeeze = graph:new_operation("Squeeze", prefix .. "/squeeze")
  squeeze:add_input(pooled)
  squeeze:set_type("T", dtype)
  squeeze:set_int_list("squeeze_dims", { 1 })
  return { output = squeeze:finish(), variables = {}, initializers = {} }
end

function Layers.avg_pool1d(graph, input, kernel_size, options)
  options = options or {}
  local kernel = kernel_size
  if type(kernel) ~= "number" or kernel < 1 then
    error("avg_pool1d requires a positive kernel size", 3)
  end
  local stride = options.stride or kernel
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "avg_pool1d"
  local expand = graph:new_operation("ExpandDims", prefix .. "/expand")
  expand:add_input(input)
  expand:add_input(tf.constant(graph, 1, prefix .. "/axis", "int32"))
  expand:set_type("T", dtype)
  expand:set_type("Tdim", "int32")
  local lifted = expand:finish()
  local pool = graph:new_operation("AvgPool", prefix .. "/pool")
  pool:add_input(lifted)
  pool:set_int_list("ksize", { 1, 1, kernel, 1 })
  pool:set_int_list("strides", { 1, 1, stride, 1 })
  pool:set_string("padding", options.padding or "VALID")
  pool:set_string("data_format", "NHWC")
  local pooled = pool:finish()
  local squeeze = graph:new_operation("Squeeze", prefix .. "/squeeze")
  squeeze:add_input(pooled)
  squeeze:set_type("T", dtype)
  squeeze:set_int_list("squeeze_dims", { 1 })
  return { output = squeeze:finish(), variables = {}, initializers = {} }
end

function Layers.global_avg_pool1d(graph, input, options)
  options = options or {}
  local tf = require("tensorflow")
  local prefix = options.name or "global_avg_pool1d"
  local pooled = tf.reduce_mean(graph, input, { 1 }, prefix .. "/mean",
    options.dtype or "float32", false)
  return { output = pooled, variables = {}, initializers = {} }
end

function Layers.separable_conv2d(graph, input, filters, kernel_size, options)
  options = options or {}
  if type(filters) ~= "number" or filters < 1 then
    error("separable_conv2d requires positive filters", 3)
  end
  local in_channels = options.in_channels
  if type(in_channels) ~= "number" or in_channels < 1 then
    error("separable_conv2d requires options.in_channels", 3)
  end
  local multiplier = options.depth_multiplier or 1
  if type(multiplier) ~= "number" or multiplier % 1 ~= 0 or multiplier < 1 then
    error("separable_conv2d depth_multiplier must be a positive integer", 3)
  end
  local kernel_h, kernel_w = kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_h, kernel_w = kernel_size[1], kernel_size[2]
  end
  local stride_h, stride_w = options.stride or 1, options.stride or 1
  if type(options.stride) == "table" then
    stride_h, stride_w = options.stride[1], options.stride[2]
  end
  local dilation_h, dilation_w = options.dilation or 1, options.dilation or 1
  if type(options.dilation) == "table" then
    dilation_h, dilation_w = options.dilation[1], options.dilation[2]
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "separable_conv2d"
  local tf = require("tensorflow")
  local depth_bound = math.sqrt(6 / (kernel_h * kernel_w * in_channels))
  local depth_values = {}
  for y = 1, kernel_h do
    depth_values[y] = {}
    for x = 1, kernel_w do
      depth_values[y][x] = {}
      for c = 1, in_channels do
        depth_values[y][x][c] = {}
        for m = 1, multiplier do
          depth_values[y][x][c][m] = (math.random() * 2 - 1) * depth_bound
        end
      end
    end
  end
  local depthwise = tf.variable(graph, depth_values, prefix .. "/depthwise",
    { dtype = dtype })
  local point_bound = math.sqrt(6 / (in_channels * multiplier + filters))
  local point_values = { { {} } }
  point_values[1][1] = {}
  for c = 1, in_channels * multiplier do
    point_values[1][1][c] = {}
    for f = 1, filters do
      point_values[1][1][c][f] = (math.random() * 2 - 1) * point_bound
    end
  end
  local pointwise = tf.variable(graph, point_values, prefix .. "/pointwise",
    { dtype = dtype })
  -- The TensorFlow C library on this host does not register the
  -- SeparableConv2D fused op, so compose the portable depthwise + pointwise
  -- pair explicitly.  Both operations are covered by TensorFlow gradients.
  local depth_desc = graph:new_operation("DepthwiseConv2dNative",
    prefix .. "/depthwise_conv")
  depth_desc:add_input(input)
  depth_desc:add_input(depthwise.read)
  depth_desc:set_type("T", dtype)
  depth_desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
  depth_desc:set_string("padding", options.padding or "SAME")
  depth_desc:set_string("data_format", "NHWC")
  depth_desc:set_int_list("dilations", { 1, dilation_h, dilation_w, 1 })
  local depth_output = depth_desc:finish()
  local point_desc = graph:new_operation("Conv2D", prefix .. "/pointwise_conv")
  point_desc:add_input(depth_output)
  point_desc:add_input(pointwise.read)
  point_desc:set_type("T", dtype)
  point_desc:set_int_list("strides", { 1, 1, 1, 1 })
  point_desc:set_string("padding", "SAME")
  point_desc:set_string("data_format", "NHWC")
  point_desc:set_int_list("dilations", { 1, 1, 1, 1 })
  point_desc:set_bool("use_cudnn_on_gpu", false)
  local output = point_desc:finish()
  local variables = { depthwise, pointwise }
  local initializers = { depthwise.initializer, pointwise.initializer }
  local bias = nil
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, filters do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  local activation = options.activation
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. activation, dtype)
  elseif type(activation) == "function" then
    output = activation(graph, output, prefix .. "/activation", dtype)
  end
  return {
    output = output, depthwise = depthwise, pointwise = pointwise, bias = bias,
    variables = variables, initializers = initializers, filters = filters,
    kernel_size = { kernel_h, kernel_w }, dtype = dtype,
  }
end

function Layers.layer_normalization(graph, input, options)
  options = options or {}
  local shape = options.normalized_shape or options.input_shape
  if type(shape) ~= "table" or #shape == 0 then
    error("layer_normalization requires options.normalized_shape", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "layer_normalization"
  local tf = require("tensorflow")
  local axes = {}
  for index = 1, #shape do axes[index] = index - #shape - 1 end
  local epsilon = options.epsilon or 1e-5
  local mean = tf.reduce_mean(graph, input, axes, prefix .. "/mean", dtype, true)
  local centered = tf.sub(graph, input, mean, prefix .. "/centered", dtype)
  local variance = tf.reduce_mean(graph, tf.square(graph, centered,
    prefix .. "/squared", dtype), axes, prefix .. "/variance", dtype, true)
  local epsilon_op = tf.constant(graph, epsilon, prefix .. "/epsilon", dtype)
  local safe_variance = tf.maximum(graph, variance, epsilon_op,
    prefix .. "/safe_variance", dtype)
  local denominator = tf.sqrt(graph, safe_variance, prefix .. "/denominator",
    dtype)
  local normalized = tf.div(graph, centered, denominator,
    prefix .. "/normalized", dtype)
  local gamma_values = {}
  local beta_values = {}
  local function build(dimension, target, value)
    if dimension > #shape then return value end
    local out = {}
    for index = 1, shape[dimension] do
      out[index] = build(dimension + 1, target, value)
    end
    return out
  end
  gamma_values = build(1, nil, 1)
  beta_values = build(1, nil, 0)
  local gamma = tf.variable(graph, gamma_values, prefix .. "/gamma",
    { dtype = dtype })
  local beta = tf.variable(graph, beta_values, prefix .. "/beta",
    { dtype = dtype })
  local output = tf.add(graph, tf.mul(graph, normalized, gamma.read,
    prefix .. "/scaled", dtype), beta.read, prefix .. "/output", dtype)
  return {
    output = output, gamma = gamma, beta = beta,
    variables = { gamma, beta },
    initializers = { gamma.initializer, beta.initializer },
    normalized_shape = shape, dtype = dtype,
  }
end

function Layers.max_pool3d(graph, input, kernel_size, options)
  options = options or {}
  local kernel_d, kernel_h, kernel_w = kernel_size, kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_d, kernel_h, kernel_w = kernel_size[1], kernel_size[2], kernel_size[3]
  end
  local stride_d, stride_h, stride_w = options.stride or kernel_d,
    options.stride or kernel_h, options.stride or kernel_w
  if type(options.stride) == "table" then
    stride_d, stride_h, stride_w = options.stride[1], options.stride[2],
      options.stride[3]
  end
  local desc = graph:new_operation("MaxPool3D", options.name or "max_pool3d")
  desc:add_input(input)
  desc:set_int_list("ksize", { 1, kernel_d, kernel_h, kernel_w, 1 })
  desc:set_int_list("strides", { 1, stride_d, stride_h, stride_w, 1 })
  desc:set_string("padding", options.padding or "VALID")
  desc:set_string("data_format", "NDHWC")
  return { output = desc:finish(), variables = {}, initializers = {} }
end

function Layers.avg_pool3d(graph, input, kernel_size, options)
  options = options or {}
  local kernel_d, kernel_h, kernel_w = kernel_size, kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_d, kernel_h, kernel_w = kernel_size[1], kernel_size[2], kernel_size[3]
  end
  local stride_d, stride_h, stride_w = options.stride or kernel_d,
    options.stride or kernel_h, options.stride or kernel_w
  if type(options.stride) == "table" then
    stride_d, stride_h, stride_w = options.stride[1], options.stride[2],
      options.stride[3]
  end
  local desc = graph:new_operation("AvgPool3D", options.name or "avg_pool3d")
  desc:add_input(input)
  desc:set_int_list("ksize", { 1, kernel_d, kernel_h, kernel_w, 1 })
  desc:set_int_list("strides", { 1, stride_d, stride_h, stride_w, 1 })
  desc:set_string("padding", options.padding or "VALID")
  desc:set_string("data_format", "NDHWC")
  return { output = desc:finish(), variables = {}, initializers = {} }
end

function Layers.global_avg_pool3d(graph, input, options)
  options = options or {}
  local tf = require("tensorflow")
  local prefix = options.name or "global_avg_pool3d"
  local pooled = tf.reduce_mean(graph, input, { 1, 2, 3 }, prefix .. "/mean",
    options.dtype or "float32", false)
  return { output = pooled, variables = {}, initializers = {} }
end

function Layers.global_max_pool3d(graph, input, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "global_max_pool3d"
  local tf = require("tensorflow")
  local desc = graph:new_operation("Max", prefix .. "/max")
  desc:add_input(input)
  desc:add_input(tf.constant(graph, { 1, 2, 3 }, prefix .. "/axes", "int32"))
  desc:set_type("T", dtype)
  desc:set_type("Tidx", "int32")
  desc:set_bool("keep_dims", false)
  return { output = desc:finish(), variables = {}, initializers = {} }
end

local function batch_matmul(graph, a, b, name, dtype, transpose_b)
  local desc = graph:new_operation("BatchMatMulV2", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  desc:set_bool("adj_x", false)
  desc:set_bool("adj_y", transpose_b == true)
  return desc:finish()
end

local function project_sequence(graph, input, input_size, time, units, name,
    dtype)
  local tf = require("tensorflow")
  local flat = tf.reshape(graph, input, { -1, input_size }, name .. "/flatten",
    dtype)
  local layer = tf.layers.dense(graph, flat, units, {
    input_size = input_size, dtype = dtype, name = name,
  })
  local output = tf.reshape(graph, layer.output, { -1, time, units },
    name .. "/sequence", dtype)
  return output, layer.variables, layer.initializers
end

local function transpose_heads(graph, input, perm, shape, name, dtype)
  local tf = require("tensorflow")
  local desc = graph:new_operation("Transpose", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, perm, name .. "/perm", "int32"))
  desc:set_type("T", dtype)
  desc:set_type("Tperm", "int32")
  return tf.reshape(graph, desc:finish(), shape, name .. "/reshape", dtype)
end

-- Scaled dot-product attention over rank-3 [batch, time, features] tensors.
-- Returns the context tensor and the attention weights.
function Layers.attention(graph, query, key, value, options)
  options = options or {}
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "attention"
  local query_shape = options.query_shape
  local key_shape = options.key_shape
  if type(query_shape) ~= "table" or #query_shape ~= 2 or
      type(key_shape) ~= "table" or #key_shape ~= 2 then
    error("attention requires query_shape and key_shape = { time, features }", 3)
  end
  local time_q, feature_q = query_shape[1], query_shape[2]
  local time_k, feature_k = key_shape[1], key_shape[2]
  if feature_q ~= feature_k then
    error("attention query and key feature sizes differ", 3)
  end
  local scores = batch_matmul(graph, query, key, prefix .. "/scores", dtype,
    true)
  local scale = 1 / math.sqrt(feature_q)
  scores = tf.mul(graph, scores, tf.constant(graph, scale, prefix .. "/scale",
    dtype), prefix .. "/scaled", dtype)
  local mask = options.mask
  if options.causal then
    local values = {}
    for i = 1, time_q do
      values[i] = {}
      for j = 1, time_k do values[i][j] = j <= i and 0 or -1e9 end
    end
    mask = tf.constant(graph, values, prefix .. "/causal", dtype)
  end
  if mask ~= nil then
    scores = tf.add(graph, scores, mask, prefix .. "/masked", dtype)
  end
  local flat = tf.reshape(graph, scores, { -1, time_k }, prefix .. "/flat",
    dtype)
  local probabilities = tf.nn.softmax(graph, flat, prefix .. "/softmax", dtype)
  local weights = tf.reshape(graph, probabilities, { -1, time_q, time_k },
    prefix .. "/weights", dtype)
  local context = batch_matmul(graph, weights, value, prefix .. "/context",
    dtype, false)
  return context, weights
end

-- Multi-head scaled dot-product attention.  Inputs are rank-3
-- [batch, time, features]; query_shape/key_shape/value_shape are required.
-- Returns the projected context and per-head attention weights.
function Layers.multi_head_attention(graph, query, key, value, options)
  options = options or {}
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "multi_head_attention"
  local heads = options.num_heads or 8
  local key_dim = options.key_dim or 64
  local value_dim = options.value_dim or key_dim
  local output_dim = options.output_dim or options.query_shape and
    options.query_shape[2] or key_dim
  if type(heads) ~= "number" or heads % 1 ~= 0 or heads < 1 then
    error("multi_head_attention num_heads must be a positive integer", 3)
  end
  local query_shape = options.query_shape
  local key_shape = options.key_shape
  local value_shape = options.value_shape or key_shape
  if type(query_shape) ~= "table" or #query_shape ~= 2 or
      type(key_shape) ~= "table" or #key_shape ~= 2 or
      type(value_shape) ~= "table" or #value_shape ~= 2 then
    error("multi_head_attention requires query/key/value shapes", 3)
  end
  local time_q, feature_q = query_shape[1], query_shape[2]
  local time_k, feature_k = key_shape[1], key_shape[2]
  local time_v, feature_v = value_shape[1], value_shape[2]
  if time_k ~= time_v then
    error("multi_head_attention key and value time sizes differ", 3)
  end

  local q, q_vars, q_init = project_sequence(graph, query, feature_q, time_q,
    heads * key_dim, prefix .. "/query", dtype)
  local k, k_vars, k_init = project_sequence(graph, key, feature_k, time_k,
    heads * key_dim, prefix .. "/key", dtype)
  local v, v_vars, v_init = project_sequence(graph, value, feature_v, time_v,
    heads * value_dim, prefix .. "/value", dtype)

  q = tf.reshape(graph, q, { -1, time_q, heads, key_dim },
    prefix .. "/query_split", dtype)
  k = tf.reshape(graph, k, { -1, time_k, heads, key_dim },
    prefix .. "/key_split", dtype)
  v = tf.reshape(graph, v, { -1, time_v, heads, value_dim },
    prefix .. "/value_split", dtype)
  q = transpose_heads(graph, q, { 0, 2, 1, 3 },
    { -1, heads, time_q, key_dim }, prefix .. "/query_heads", dtype)
  k = transpose_heads(graph, k, { 0, 2, 1, 3 },
    { -1, heads, time_k, key_dim }, prefix .. "/key_heads", dtype)
  v = transpose_heads(graph, v, { 0, 2, 1, 3 },
    { -1, heads, time_v, value_dim }, prefix .. "/value_heads", dtype)

  local scores = batch_matmul(graph, q, k, prefix .. "/scores", dtype, true)
  local scale = 1 / math.sqrt(key_dim)
  scores = tf.mul(graph, scores, tf.constant(graph, scale, prefix .. "/scale",
    dtype), prefix .. "/scaled", dtype)
  local mask = options.mask
  if options.causal then
    local values = {}
    for i = 1, time_q do
      values[i] = {}
      for j = 1, time_k do values[i][j] = j <= i and 0 or -1e9 end
    end
    mask = tf.constant(graph, values, prefix .. "/causal", dtype)
  end
  if mask ~= nil then
    scores = tf.add(graph, scores, mask, prefix .. "/masked", dtype)
  end
  local flat = tf.reshape(graph, scores, { -1, time_k }, prefix .. "/flat",
    dtype)
  local probabilities = tf.nn.softmax(graph, flat, prefix .. "/softmax", dtype)
  local weights = tf.reshape(graph, probabilities,
    { -1, heads, time_q, time_k }, prefix .. "/weights", dtype)
  local context = batch_matmul(graph, weights, v, prefix .. "/context", dtype,
    false)
  context = transpose_heads(graph, context, { 0, 2, 1, 3 },
    { -1, time_q, heads * value_dim }, prefix .. "/context_heads", dtype)
  local output, o_vars, o_init = project_sequence(graph, context,
    heads * value_dim, time_q, output_dim, prefix .. "/output", dtype)
  local variables, initializers = {}, {}
  for _, group in ipairs({ q_vars, k_vars, v_vars, o_vars }) do
    for _, variable in ipairs(group) do variables[#variables + 1] = variable end
  end
  for _, group in ipairs({ q_init, k_init, v_init, o_init }) do
    for _, initializer in ipairs(group) do
      initializers[#initializers + 1] = initializer
    end
  end
  return {
    output = output, weights = weights, variables = variables,
    initializers = initializers, num_heads = heads, key_dim = key_dim,
    value_dim = value_dim, output_dim = output_dim, dtype = dtype,
  }
end

function Layers.activation(graph, input, activation, options)
  options = options or {}
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "activation"
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    return { output = fn(graph, input, prefix, dtype), variables = {},
      initializers = {} }
  elseif type(activation) == "function" then
    return { output = activation(graph, input, prefix, dtype),
      variables = {}, initializers = {} }
  end
  error("activation layer requires an activation name or function", 3)
end

function Layers.flatten(graph, input, input_size, options)
  options = options or {}
  if type(input_size) ~= "number" or input_size < 1 then
    error("flatten requires a positive input_size", 3)
  end
  local tf = require("tensorflow")
  local output = tf.reshape(graph, input, { -1, input_size },
    options.name or "flatten", options.dtype or "float32")
  return {
    output = output,
    input_size = input_size,
    variables = {},
    initializers = {},
    dtype = options.dtype or "float32",
  }
end

function Layers.reshape(graph, input, shape, options)
  options = options or {}
  if type(shape) ~= "table" or #shape == 0 then
    error("reshape requires a non-empty target shape", 3)
  end
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "reshape"
  local desc = graph:new_operation("Reshape", prefix .. "/reshape")
  desc:add_input(input)
  desc:add_input(tf.constant(graph, shape, prefix .. "/shape", "int32"))
  desc:set_type("T", dtype)
  desc:set_type("Tshape", "int32")
  return { output = desc:finish(), variables = {}, initializers = {},
    shape = shape }
end

function Layers.permute(graph, input, perm, options)
  options = options or {}
  if type(perm) ~= "table" or #perm == 0 then
    error("permute requires a permutation list", 3)
  end
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "permute"
  local desc = graph:new_operation("Transpose", prefix .. "/transpose")
  desc:add_input(input)
  desc:add_input(tf.constant(graph, perm, prefix .. "/perm", "int32"))
  desc:set_type("T", dtype)
  desc:set_type("Tperm", "int32")
  return { output = desc:finish(), variables = {}, initializers = {},
    perm = perm }
end

function Layers.upsampling2d(graph, input, scale, options)
  options = options or {}
  local scale_h, scale_w = scale, scale
  if type(scale) == "table" then scale_h, scale_w = scale[1], scale[2] end
  if type(scale_h) ~= "number" or type(scale_w) ~= "number" or
      scale_h < 1 or scale_w < 1 then
    error("upsampling2d requires a positive scale", 3)
  end
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "upsampling2d"
  local input_shape = options.input_shape
  if type(input_shape) ~= "table" or #input_shape ~= 3 then
    error("upsampling2d requires options.input_shape = {h, w, c}", 3)
  end
  local size = tf.constant(graph,
    { math.floor(input_shape[1] * scale_h), math.floor(input_shape[2] * scale_w) },
    prefix .. "/size", "int32")
  local desc = graph:new_operation("ResizeNearestNeighbor",
    prefix .. "/resize")
  desc:add_input(input)
  desc:add_input(size)
  desc:set_type("T", dtype)
  desc:set_bool("align_corners", false)
  desc:set_bool("half_pixel_centers", false)
  return { output = desc:finish(), variables = {}, initializers = {},
    scale = { scale_h, scale_w } }
end

function Layers.zero_padding2d(graph, input, padding, options)
  options = options or {}
  local top, bottom, left, right
  if type(padding) == "number" then
    top, bottom, left, right = padding, padding, padding, padding
  elseif type(padding) == "table" then
    top, bottom, left, right = padding[1], padding[2], padding[3], padding[4]
  end
  if not top or not bottom or not left or not right then
    error("zero_padding2d requires a number or {top, bottom, left, right}", 3)
  end
  local tf = require("tensorflow")
  local dtype = options.dtype or "float32"
  local prefix = options.name or "zero_padding2d"
  local paddings = tf.constant(graph,
    { { 0, 0 }, { top, bottom }, { left, right }, { 0, 0 } },
    prefix .. "/paddings", "int32")
  local desc = graph:new_operation("Pad", prefix .. "/pad")
  desc:add_input(input)
  desc:add_input(paddings)
  desc:set_type("T", dtype)
  desc:set_type("Tpaddings", "int32")
  return { output = desc:finish(), variables = {}, initializers = {},
    padding = { top, bottom, left, right } }
end

function Layers.depthwise_conv2d(graph, input, multiplier, kernel_size, options)
  options = options or {}
  local in_channels = options.in_channels
  if type(in_channels) ~= "number" or in_channels < 1 then
    error("depthwise_conv2d requires options.in_channels", 3)
  end
  multiplier = multiplier or 1
  local kernel_h, kernel_w = kernel_size, kernel_size
  if type(kernel_size) == "table" then
    kernel_h, kernel_w = kernel_size[1], kernel_size[2]
  end
  local stride_h, stride_w = options.stride or 1, options.stride or 1
  if type(options.stride) == "table" then
    stride_h, stride_w = options.stride[1], options.stride[2]
  end
  local dilation_h, dilation_w = options.dilation or 1, options.dilation or 1
  if type(options.dilation) == "table" then
    dilation_h, dilation_w = options.dilation[1], options.dilation[2]
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "depthwise_conv2d"
  local tf = require("tensorflow")
  local bound = math.sqrt(6 / (kernel_h * kernel_w * in_channels))
  local values = {}
  for y = 1, kernel_h do
    values[y] = {}
    for x = 1, kernel_w do
      values[y][x] = {}
      for c = 1, in_channels do
        values[y][x][c] = {}
        for m = 1, multiplier do
          values[y][x][c][m] = (math.random() * 2 - 1) * bound
        end
      end
    end
  end
  local weight = tf.variable(graph, values, prefix .. "/weight",
    { dtype = dtype })
  local desc = graph:new_operation("DepthwiseConv2dNative",
    prefix .. "/conv")
  desc:add_input(input)
  desc:add_input(weight.read)
  desc:set_type("T", dtype)
  desc:set_int_list("strides", { 1, stride_h, stride_w, 1 })
  desc:set_string("padding", options.padding or "SAME")
  desc:set_string("data_format", "NHWC")
  desc:set_int_list("dilations", { 1, dilation_h, dilation_w, 1 })
  local output = desc:finish()
  local variables, initializers = { weight }, { weight.initializer }
  local bias = nil
  local channels = in_channels * multiplier
  if options.use_bias ~= false then
    local zeros = {}
    for i = 1, channels do zeros[i] = 0 end
    bias = tf.variable(graph, zeros, prefix .. "/bias", { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
    output = tf.add(graph, output, bias.read, prefix .. "/bias_add", dtype)
  end
  local activation = options.activation
  if type(activation) == "string" then
    local fn = tf.nn[activation]
    if fn == nil then error("unknown activation '" .. activation .. "'", 3) end
    output = fn(graph, output, prefix .. "/" .. activation, dtype)
  elseif type(activation) == "function" then
    output = activation(graph, output, prefix .. "/activation", dtype)
  end
  return {
    output = output, weight = weight, bias = bias, variables = variables,
    initializers = initializers, channels = channels,
    kernel_size = { kernel_h, kernel_w }, dtype = dtype,
  }
end

function Layers.global_max_pool1d(graph, input, options)
  options = options or {}
  local dtype = options.dtype or "float32"
  local prefix = options.name or "global_max_pool1d"
  local desc = graph:new_operation("Max", prefix .. "/max")
  desc:add_input(input)
  desc:add_input(require("tensorflow").constant(graph, { 1 },
    prefix .. "/axes", "int32"))
  desc:set_type("T", dtype)
  desc:set_type("Tidx", "int32")
  desc:set_bool("keep_dims", false)
  return { output = desc:finish(), variables = {}, initializers = {} }
end

function Layers.dropout(graph, input, rate, options)
  options = options or {}
  rate = rate or 0.5
  if type(rate) ~= "number" or rate < 0 or rate >= 1 then
    error("dropout rate must be in [0, 1)", 3)
  end
  local training = options.training
  if training == false then return { output = input } end
  local tf = require("tensorflow")
  local prefix = options.name or "dropout"
  local dtype = options.dtype or "float32"
  local keep = 1 - rate
  local shape = tf.shape(graph, input, prefix .. "/shape")
  local random = graph:new_operation("RandomUniform", prefix .. "/random")
  random:add_input(shape)
  random:set_type("dtype", dtype)
  random:set_int("seed", options.seed or 0)
  random:set_int("seed2", options.seed2 or 0)
  local noise = random:finish()
  local threshold = tf.constant(graph, keep, prefix .. "/keep", dtype)
  local less = graph:new_operation("Less", prefix .. "/mask")
  less:add_input(noise)
  less:add_input(threshold)
  less:set_type("T", dtype)
  local mask = less:finish()
  local cast = graph:new_operation("Cast", prefix .. "/cast")
  cast:add_input(mask)
  cast:set_type("SrcT", "bool")
  cast:set_type("DstT", dtype)
  local scale = cast:finish()
  local scaled = tf.mul(graph, input, scale, prefix .. "/scale", dtype)
  local dropped = tf.div(graph, scaled, tf.constant(graph, keep,
    prefix .. "/divide_by", dtype), prefix .. "/dropped", dtype)
  if type(training) == "table" and training.oper ~= nil then
    local cast_training = graph:new_operation("Cast", prefix .. "/training")
    cast_training:add_input(training)
    cast_training:set_type("SrcT", "bool")
    cast_training:set_type("DstT", dtype)
    local gate = cast_training:finish()
    local delta = tf.sub(graph, dropped, input, prefix .. "/delta", dtype)
    local gated = tf.mul(graph, gate, delta, prefix .. "/gate", dtype)
    return { output = tf.add(graph, input, gated, prefix .. "/output", dtype) }
  end
  return { output = dropped }
end
function Layers.embedding(graph, input, input_dim, output_dim, options)
  options = options or {}
  if type(input_dim) ~= "number" or type(output_dim) ~= "number" or
      input_dim < 1 or output_dim < 1 then
    error("embedding requires positive input and output dimensions", 3)
  end
  local tf = require("tensorflow")
  local prefix = options.name or "embedding"
  local dtype = options.dtype or "float32"
  local indices_dtype = options.indices_dtype or "int32"
  local bound = 1 / math.sqrt(output_dim)
  local values = random_uniform(input_dim, output_dim, bound)
  if options.padding_idx ~= nil then
    local padding = options.padding_idx
    if type(padding) ~= "number" or padding % 1 ~= 0 or
        padding < 0 or padding >= input_dim then
      error("embedding padding_idx must be an integer in [0, input_dim)", 3)
    end
    for column = 1, output_dim do values[padding + 1][column] = 0 end
  end
  local weight = tf.variable(graph, values, prefix .. "/weight",
    { dtype = dtype })
  local axis = tf.constant(graph, 0, prefix .. "/axis", "int32")
  local desc = graph:new_operation("GatherV2", prefix .. "/gather")
  desc:add_input(weight.read)
  desc:add_input(input)
  desc:add_input(axis)
  desc:set_type("Tparams", dtype)
  desc:set_type("Tindices", indices_dtype)
  desc:set_int("batch_dims", 0)
  return {
    output = desc:finish(),
    weight = weight,
    variables = { weight },
    initializers = { weight.initializer },
    input_dim = input_dim,
    output_dim = output_dim,
    dtype = dtype,
  }
end

function Layers.sequential(graph, input, definitions, input_size)
  local layers = {}
  for i, definition in ipairs(definitions) do
    local options = definition.options or {}
    options.input_size = options.input_size or
      (i == 1 and input_size or layers[i - 1].units)
    options.name = options.name or ("layer_" .. i)
    local layer = Layers.dense(graph, input, definition.units, options)
    layers[#layers + 1] = layer
    input = layer.output
  end
  return layers
end

return Layers
