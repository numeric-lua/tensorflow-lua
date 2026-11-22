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

function Graph:output_spec(output)
  local lib = capi.load()
  local operation = output.oper or output
  return ffi.string(lib.TF_OperationName(operation)) .. ":" ..
    tostring(output.index or 0)
end

function Graph:new_operation(op_type, name)
  return require("tensorflow.graph_builder").new_operation(self, op_type, name)
end

function Graph:import_graph_def(data, options)
  if type(data) ~= "string" then
    error("import_graph_def expects serialized GraphDef bytes", 2)
  end
  if options ~= nil and type(options) ~= "table" then
    error("import_graph_def options must be a table", 2)
  end
  local prefix = options and options.prefix
  if prefix ~= nil and type(prefix) ~= "string" then
    error("GraphDef import prefix must be a string", 2)
  end
  local lib = capi.load()
  local buffer = ffi.gc(capi.require(
    lib.TF_NewBufferFromString(data, #data), "TF_NewBufferFromString"),
    function(value) lib.TF_DeleteBuffer(value) end)
  local import_options = ffi.gc(capi.require(
    lib.TF_NewImportGraphDefOptions(), "TF_NewImportGraphDefOptions"),
    function(value) lib.TF_DeleteImportGraphDefOptions(value) end)
  if prefix ~= nil then
    lib.TF_ImportGraphDefOptionsSetPrefix(import_options, prefix)
  end
  local status = capi.status()
  lib.TF_GraphImportGraphDef(self:handle(), buffer, import_options, status)
  local ok, err = pcall(capi.check, status, "GraphDef import")
  ffi.gc(buffer, nil)
  lib.TF_DeleteBuffer(buffer)
  ffi.gc(import_options, nil)
  lib.TF_DeleteImportGraphDefOptions(import_options)
  if not ok then error(err, 2) end
  return self
end

function Graph:import_file(path, options)
  local file, err = io.open(path, "rb")
  if file == nil then error("cannot open GraphDef '" .. path .. "': " .. tostring(err), 2) end
  local data = file:read("*a")
  file:close()
  return self:import_graph_def(data, options)
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
