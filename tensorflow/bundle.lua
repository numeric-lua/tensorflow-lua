--
-- Portable deployment bundle: GraphDef + native V2 checkpoint + manifest.
--
-- This is not official SavedModel export (the C API has no MetaGraphDef
-- writer).  It is a self-contained CPU deployment format for the Lua runtime.
--
local Bundle = {}

local function ensure_directory(path)
  local status = os.execute("mkdir -p '" .. path:gsub("'", "'\\''") .. "'")
  if status ~= 0 and status ~= true then
    error("cannot create bundle directory '" .. path .. "'", 3)
  end
end

local function write_file(path, contents)
  local file, err = io.open(path, "wb")
  if file == nil then error("cannot write '" .. path .. "': " .. tostring(err), 3) end
  local ok, write_err = file:write(contents)
  file:close()
  if ok == nil then error("write failed: " .. tostring(write_err), 3) end
end

function Bundle.save(directory, graph, session, variables, metadata)
  if type(directory) ~= "string" or directory == "" then
    error("bundle.save expects a directory", 2)
  end
  if type(variables) ~= "table" or #variables == 0 then
    error("bundle.save expects variables", 2)
  end
  metadata = metadata or {}
  ensure_directory(directory)
  write_file(directory .. "/graph.pb", graph:graph_def())
  local tf = require("tensorflow")
  local entries = {}
  local manifest_variables = {}
  for index, entry in ipairs(variables) do
    local variable = entry.variable or entry
    local name = entry.name or variable.name
    if name == nil then error("bundle variable " .. index .. " has no name", 2) end
    entries[#entries + 1] = { name = name, variable = variable }
    manifest_variables[#manifest_variables + 1] = {
      name = name, dtype = variable.dtype or entry.dtype or "float32",
      shape = variable.shape or entry.shape,
    }
  end
  session:run({}, {}, { tf.save_v2(graph, directory .. "/variables", entries) })
  local manifest = {
    format = "tensorflow-lua-bundle",
    version = metadata.version or require("tensorflow").VERSION,
    inputs = metadata.inputs or {},
    outputs = metadata.outputs or {},
    variables = manifest_variables,
  }
  write_file(directory .. "/manifest.lua",
    require("tensorflow.serialize").encode(manifest))
  return directory
end

function Bundle.load(directory)
  if type(directory) ~= "string" or directory == "" then
    error("bundle.load expects a directory", 2)
  end
  local graph_file, err = io.open(directory .. "/graph.pb", "rb")
  if graph_file == nil then
    error("cannot read bundle graph: " .. tostring(err), 2)
  end
  local graph_bytes = graph_file:read("*a")
  graph_file:close()
  local manifest_file, manifest_err = io.open(directory .. "/manifest.lua", "rb")
  if manifest_file == nil then
    error("cannot read bundle manifest: " .. tostring(manifest_err), 2)
  end
  local manifest_bytes = manifest_file:read("*a")
  manifest_file:close()
  local manifest = require("tensorflow.serialize").decode(manifest_bytes)
  if manifest.format ~= "tensorflow-lua-bundle" then
    error("not a tensorflow-lua bundle", 2)
  end
  local tf = require("tensorflow")
  local graph = tf.Graph.new()
  graph:import_graph_def(graph_bytes)
  local session = tf.Session.new(graph)
  local entries = {}
  for _, spec in ipairs(manifest.variables or {}) do
    entries[#entries + 1] = {
      name = spec.name, variable = spec.name .. ":0", dtype = spec.dtype,
      shape = spec.shape,
    }
  end
  if manifest.variables ~= nil then
    session:run({}, {}, tf.restore_v2(graph, directory .. "/variables", entries))
  end
  return {
    graph = graph, session = session, manifest = manifest,
    inputs = manifest.inputs, outputs = manifest.outputs,
  }
end

return Bundle
