--
-- Minimal protobuf wire decoder for SavedModel MetaGraphDef signatures.
--
local Proto = {}

local function varint(data, position)
  local value, shift = 0, 0
  while position <= #data do
    local byte = data:byte(position)
    position = position + 1
    value = value + (byte % 128) * 2 ^ shift
    if byte < 128 then return value, position end
    shift = shift + 7
    if shift > 63 then error("protobuf varint overflow", 3) end
  end
  error("truncated protobuf varint", 3)
end

local function fields(data)
  local result = {}
  local position = 1
  while position <= #data do
    local tag, next_position = varint(data, position)
    position = next_position
    local number = math.floor(tag / 8)
    local wire = tag % 8
    if number == 0 then error("protobuf field number 0", 3) end
    local value
    if wire == 0 then
      value, position = varint(data, position)
    elseif wire == 1 then
      value = data:sub(position, position + 7)
      position = position + 8
    elseif wire == 2 then
      local length
      length, position = varint(data, position)
      if length > #data - position + 1 then
        error("truncated protobuf field", 3)
      end
      value = data:sub(position, position + length - 1)
      position = position + length
    elseif wire == 5 then
      value = data:sub(position, position + 3)
      position = position + 4
    else
      error("unsupported protobuf wire type " .. tostring(wire), 3)
    end
    result[#result + 1] = { number = number, wire = wire, value = value }
  end
  return result
end

local function tensor_info(data)
  local info = { dtype = 0 }
  for _, field in ipairs(fields(data)) do
    if field.number == 1 then
      if field.wire ~= 2 then error("TensorInfo.name is not a string", 3) end
      info.name = field.value
    elseif field.number == 2 then info.dtype = field.value end
  end
  return info
end

local function tensor_map(data)
  local key, value
  for _, field in ipairs(fields(data)) do
    if field.number == 1 then
      if field.wire ~= 2 then error("map key is not a string", 3) end
      key = field.value
    elseif field.number == 2 then value = tensor_info(field.value) end
  end
  return key, value
end

local function signature_def(data)
  local signature = { inputs = {}, outputs = {}, method_name = "" }
  for _, field in ipairs(fields(data)) do
    if field.number == 1 or field.number == 2 then
      local key, info = tensor_map(field.value)
      if key and info then
        if field.number == 1 then signature.inputs[key] = info
        else signature.outputs[key] = info end
      end
    elseif field.number == 3 then
      signature.method_name = field.value
    end
  end
  return signature
end

-- Returns `{[signature_name] = {inputs=..., outputs=..., method_name=...}}`.
function Proto.saved_model_signatures(meta_graph_def)
  local signatures = {}
  for _, field in ipairs(fields(meta_graph_def)) do
    if field.number == 5 then
      local name, value
      for _, entry in ipairs(fields(field.value)) do
        if entry.number == 1 then
          if entry.wire ~= 2 then error("signature name is not a string", 3) end
          name = entry.value
        elseif entry.number == 2 then value = signature_def(entry.value) end
      end
      if name and value then signatures[name] = value end
    end
  end
  return signatures
end

Proto.fields = fields
Proto.varint = varint

return Proto
