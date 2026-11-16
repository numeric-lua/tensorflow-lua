--
-- Small compositional layer builders on top of graph variables.
--
local Tensor = require("tensorflow.tensor")

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
    bias = tf.variable(graph, { 0 }, prefix .. "/bias", { dtype = dtype })
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
