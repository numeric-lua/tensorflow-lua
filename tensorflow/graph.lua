--
-- TF_Graph wrapper and GraphDef import.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")

local Graph = {}
Graph.__index = Graph

function Graph.new()
  local lib = capi.load()
  return setmetatable({ _ptr = lib.TF_NewGraph() }, Graph)
end

function Graph:handle()
  return self._ptr
end

function Graph:operation(name)
  local lib = capi.load()
  if type(name) ~= "string" then error("operation name must be a string", 2) end
  local op = lib.TF_GraphOperationByName(self._ptr, name)
  if op == nil then error("graph has no operation named '" .. name .. "'", 2) end
  return op
end

function Graph:import_graph_def(data)
  if type(data) ~= "string" then
    error("import_graph_def expects serialized GraphDef bytes", 2)
  end
  local lib = capi.load()
  local buffer = lib.TF_NewBufferFromString(data, #data)
  if buffer == nil then error("TF_NewBufferFromString failed", 2) end
  local options = lib.TF_NewImportGraphDefOptions()
  local status = capi.status()
  lib.TF_GraphImportGraphDef(self._ptr, buffer, options, status)
  local ok, err = pcall(capi.check, status, "GraphDef import")
  lib.TF_DeleteImportGraphDefOptions(options)
  lib.TF_DeleteBuffer(buffer)
  if not ok then error(err, 2) end
  return self
end

function Graph:import_file(path)
  local file, err = io.open(path, "rb")
  if file == nil then error("cannot open GraphDef '" .. path .. "': " .. tostring(err), 2) end
  local data = file:read("*a")
  file:close()
  return self:import_graph_def(data)
end

function Graph:graph_def()
  local lib = capi.load()
  local buffer = lib.TF_NewBuffer()
  local status = capi.status()
  lib.TF_GraphToGraphDef(self._ptr, buffer, status)
  local ok, err = pcall(capi.check, status, "GraphDef export")
  if not ok then lib.TF_DeleteBuffer(buffer); error(err, 2) end
  local struct = lib.TF_GetBuffer(buffer)
  local data = ffi.string(struct.data, struct.length)
  lib.TF_DeleteBuffer(buffer)
  return data
end

function Graph:__gc()
  if self._ptr ~= nil then
    capi.load().TF_DeleteGraph(self._ptr)
    self._ptr = nil
  end
end

return Graph
