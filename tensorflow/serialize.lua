--
-- Small Lua literal serializer for model configuration. Only data is allowed;
-- functions, userdata, and cycles are rejected. load() executes Lua from a
-- trusted config file.
--
local Serialize = {}

local function encode_value(value, seen, indent)
  local kind = type(value)
  if kind == "nil" then return "nil" end
  if kind == "boolean" then return value and "true" or "false" end
  if kind == "number" then
    if value ~= value or value == math.huge or value == -math.huge then
      error("configuration cannot contain NaN or infinity", 3)
    end
    return string.format("%.17g", value)
  end
  if kind == "string" then return string.format("%q", value) end
  if kind ~= "table" then
    error("configuration cannot contain " .. kind, 3)
  end
  if seen[value] then error("configuration cannot contain cycles", 3) end
  seen[value] = true
  local keys = {}
  local count = 0
  for key in pairs(value) do
    count = count + 1
    keys[#keys + 1] = key
  end
  table.sort(keys, function(a, b)
    local ta, tb = type(a), type(b)
    if ta ~= tb then return ta < tb end
    return tostring(a) < tostring(b)
  end)
  local parts = {}
  for _, key in ipairs(keys) do
    local key_text
    if type(key) == "string" then
      key_text = string.format("[%q]", key)
    elseif type(key) == "number" then
      key_text = "[" .. string.format("%.17g", key) .. "]"
    else
      error("configuration keys must be strings or numbers", 3)
    end
    parts[#parts + 1] = key_text .. "=" ..
      encode_value(value[key], seen, indent)
  end
  seen[value] = nil
  return "{" .. table.concat(parts, ",") .. "}"
end

function Serialize.encode(value)
  return "return " .. encode_value(value, {}, 0) .. "\n"
end

function Serialize.decode(text)
  local chunk, err = loadstring(text, "@model-config")
  if chunk == nil then error("invalid model configuration: " .. tostring(err), 2) end
  local ok, value = pcall(chunk)
  if not ok then error("invalid model configuration: " .. tostring(value), 2) end
  if type(value) ~= "table" then
    error("model configuration must return a table", 2)
  end
  return value
end

function Serialize.copy(value, seen)
  if type(value) ~= "table" then return value end
  seen = seen or {}
  if seen[value] then return seen[value] end
  local copy = {}
  seen[value] = copy
  for key, child in pairs(value) do copy[key] = Serialize.copy(child, seen) end
  return copy
end

return Serialize
