--
-- TF_Graph wrapper and GraphDef import.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")

local Graph = {}
Graph.__index = Graph

function Graph.new()
  local lib = capi.load()
  local pointer = capi.require(lib.TF_NewGraph(), "TF_NewGraph")
  local handle = ffi.gc(pointer, function(value)
    capi.release_graph(value)
  end)
  return setmetatable({ _handle = handle }, Graph)
end

function Graph:handle()
  if self._handle == nil then error("graph has been deleted", 2) end
  return self._handle
end

function Graph:delete()
  if self._handle ~= nil then
    local handle = self._handle
    self._handle = nil
    ffi.gc(handle, nil)
    capi.release_graph(handle)
  end
end

function Graph:operation(name)
  local lib = capi.load()
  if type(name) ~= "string" then error("operation name must be a string", 2) end
  local op = lib.TF_GraphOperationByName(self:handle(), name)
  if op == nil then error("graph has no operation named '" .. name .. "'", 2) end
  return op
end

function Graph:new_operation(op_type, name)
  return require("tensorflow.graph_builder").new_operation(self, op_type, name)
end

function Graph:import_graph_def(data)
  if type(data) ~= "string" then
    error("import_graph_def expects serialized GraphDef bytes", 2)
  end
  local lib = capi.load()
  local buffer = capi.require(lib.TF_NewBufferFromString(data, #data),
    "TF_NewBufferFromString")
  local options = capi.require(lib.TF_NewImportGraphDefOptions(),
    "TF_NewImportGraphDefOptions")
  local status = capi.status()
  lib.TF_GraphImportGraphDef(self:handle(), buffer, options, status)
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
  local buffer = capi.require(lib.TF_NewBuffer(), "TF_NewBuffer")
  local status = capi.status()
  lib.TF_GraphToGraphDef(self:handle(), buffer, status)
  local ok, err = pcall(capi.check, status, "GraphDef export")
  if not ok then lib.TF_DeleteBuffer(buffer); error(err, 2) end
  local struct = lib.TF_GetBuffer(buffer)
  if struct.length > 0 and struct.data == nil then
    lib.TF_DeleteBuffer(buffer)
    error("TF_GraphToGraphDef returned no data", 2)
  end
  local data = struct.length > 0 and ffi.string(struct.data, struct.length) or ""
  lib.TF_DeleteBuffer(buffer)
  return data
end



return Graph
