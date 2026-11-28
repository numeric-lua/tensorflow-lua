--
-- Reusable native V2 checkpoint saver.
--
local Saver = {}
Saver.__index = Saver

local function normalize(variables)
  local entries, seen = {}, {}
  for i, entry in ipairs(variables) do
    local variable = entry.variable or entry
    local name = entry.name or variable.name
    if type(name) ~= "string" or name == "" then
      error("Saver variable " .. i .. " is missing a name", 3)
    end
    if seen[name] then
      error("Saver variable name '" .. name .. "' is duplicated", 3)
    end
    seen[name] = true
    entries[i] = { name = name, variable = variable }
  end
  return entries
end

function Saver.new(graph, variables)
  if type(variables) ~= "table" or #variables == 0 then
    error("Saver.new expects a non-empty variable list", 2)
  end
  return setmetatable({ graph = graph, entries = normalize(variables) },
    Saver)
end

function Saver:size()
  return #self.entries
end

function Saver:names()
  local result = {}
  for i, entry in ipairs(self.entries) do result[i] = entry.name end
  return result
end

function Saver:has(name)
  for _, entry in ipairs(self.entries) do
    if entry.name == name then return true end
  end
  return false
end

local function describe(entry)
  local variable = entry.variable
  local shape = {}
  for i, dimension in ipairs(variable.shape or { }) do
    shape[i] = dimension
  end
  return { name = entry.name, dtype = variable.dtype or "float32",
    shape = shape }
end

function Saver:metadata()
  local result = {}
  for i, entry in ipairs(self.entries) do result[i] = describe(entry) end
  return result
end

function Saver:entry(name)
  for _, entry in ipairs(self.entries) do
    if entry.name == name then return describe(entry) end
  end
  return nil
end

function Saver:variable(name)
  for _, entry in ipairs(self.entries) do
    if entry.name == name then return entry.variable end
  end
  return nil
end

function Saver:save(session, prefix, name)
  local tf = require("tensorflow")
  local target = tf.save_v2(self.graph, prefix, self.entries, name)
  session:run({}, {}, { target })
  return prefix
end

function Saver:restore(session, prefix, name)
  local tf = require("tensorflow")
  local targets = tf.restore_v2(self.graph, prefix, self.entries, name)
  session:run({}, {}, targets)
  return self
end

return Saver
