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
  local weight = tf.variable(graph,
    random_uniform(input_dim, output_dim, bound), prefix .. "/weight",
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
