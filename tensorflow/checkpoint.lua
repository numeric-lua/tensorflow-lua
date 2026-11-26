--
-- Portable tensor checkpoints for tensorflow-lua variables.
--
local Tensor = require("tensorflow.tensor")

local Checkpoint = {}
local MAGIC = "TFLUACPK"
local VERSION = 1

local function u32(value)
  if value < 0 or value > 0xffffffff or value % 1 ~= 0 then
    error("checkpoint value out of uint32 range: " .. tostring(value), 3)
  end
  local bytes = {}
  for _ = 1, 4 do
    bytes[#bytes + 1] = string.char(value % 256)
    value = math.floor(value / 256)
  end
  return table.concat(bytes)
end

local function read_u32(data, position)
  if position + 3 > #data then error("truncated checkpoint", 3) end
  local value = 0
  for i = 3, 0, -1 do value = value * 256 + data:byte(position + i) end
  return value, position + 4
end

local function read_bytes(data, position, length)
  if length < 0 or position + length - 1 > #data then
    error("truncated checkpoint", 3)
  end
  return data:sub(position, position + length - 1), position + length
end

function Checkpoint.save(session, path, variables)
  if type(variables) ~= "table" or #variables == 0 then
    error("save_checkpoint expects a non-empty variable list", 3)
  end
  local fetch, names = {}, {}
  for i, entry in ipairs(variables) do
    local variable = entry.variable or entry
    names[i] = entry.name or variable.name
    if names[i] == nil then
      error("checkpoint variable is missing a name", 3)
    end
    fetch[i] = variable.read
  end
  local values = session:run({}, fetch)
  local parts = { MAGIC, u32(VERSION), u32(#values) }
  for i, tensor in ipairs(values) do
    local name = names[i]
    local shape = tensor:shape()
    local raw = tensor:bytes()
    parts[#parts + 1] = u32(#name)
    parts[#parts + 1] = name
    local _, dtype = Tensor.dtype_info(tensor:dtype())
    parts[#parts + 1] = u32(dtype.code)
    parts[#parts + 1] = u32(#shape)
    for _, dimension in ipairs(shape) do parts[#parts + 1] = u32(dimension) end
    parts[#parts + 1] = u32(#raw)
    parts[#parts + 1] = raw
  end
  local file, err = io.open(path, "wb")
  if file == nil then error("cannot open checkpoint: " .. tostring(err), 2) end
  local ok, write_err = pcall(function() file:write(table.concat(parts)) end)
  file:close()
  if not ok then error(write_err, 2) end
  return path
end

local function read_entries(path)
  local file, err = io.open(path, "rb")
  if file == nil then error("cannot open checkpoint: " .. tostring(err), 2) end
  local data = file:read("*a")
  file:close()
  if data:sub(1, #MAGIC) ~= MAGIC then error("not a tensorflow-lua checkpoint", 2) end
  local position = #MAGIC + 1
  local version
  version, position = read_u32(data, position)
  if version ~= VERSION then error("unsupported checkpoint version", 2) end
  local count
  count, position = read_u32(data, position)
  local entries = {}
  for _ = 1, count do
    local name_length
    name_length, position = read_u32(data, position)
    local name
    name, position = read_bytes(data, position, name_length)
    local dtype_code
    dtype_code, position = read_u32(data, position)
    local ndim
    ndim, position = read_u32(data, position)
    local shape = {}
    for i = 1, ndim do
      local dimension
      dimension, position = read_u32(data, position)
      shape[i] = dimension
    end
    local length
    length, position = read_u32(data, position)
    local bytes
    bytes, position = read_bytes(data, position, length)
    entries[name] = { dtype = dtype_code, shape = shape, bytes = bytes }
  end
  return entries
end

function Checkpoint.metadata(path)
  local entries = read_entries(path)
  local result = {}
  for name, entry in pairs(entries) do
    local shape = {}
    for i, dimension in ipairs(entry.shape) do shape[i] = dimension end
    result[name] = {
      dtype = Tensor.dtype_info(entry.dtype),
      shape = shape,
      bytes = #entry.bytes,
    }
  end
  return result
end

function Checkpoint.load(graph, path, variables)
  local entries = read_entries(path)
  local targets = {}
  for i, entry in ipairs(variables) do
    local variable = entry.variable or entry
    local name = entry.name or variable.name
    local saved = entries[name]
    if saved == nil then error("checkpoint has no variable '" .. name .. "'", 2) end
    local saved_dtype = Tensor.dtype_info(saved.dtype)
    if variable.dtype and saved_dtype ~= variable.dtype then
      error("checkpoint dtype mismatch for '" .. name .. "'", 2)
    end
    if type(variable.shape) == "table" then
      local compatible = #variable.shape == #saved.shape
      if compatible then
        for dimension = 1, #saved.shape do
          if variable.shape[dimension] ~= saved.shape[dimension] then
            compatible = false
            break
          end
        end
      end
      if not compatible then
        error("checkpoint shape mismatch for '" .. name .. "'", 2)
      end
    end
    local tensor = Tensor.from_bytes(saved.dtype, saved.shape, saved.bytes)
    local desc = graph:new_operation("Assign", name .. "/restore_" .. i)
    desc:add_input(variable.ref or variable)
    desc:add_input(require("tensorflow").constant(graph, tensor,
      name .. "/restore_value_" .. i))
    desc:set_type("T", variable.dtype or tensor:dtype())
    desc:set_bool("use_locking", true)
    desc:set_bool("validate_shape", true)
    targets[#targets + 1] = desc:finish()
  end
  return targets
end

return Checkpoint
