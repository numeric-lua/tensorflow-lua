--
-- TF_Graph wrapper and GraphDef import.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")

local Graph = {}
Graph.__index = Graph

local function resolve_operation(graph, value)
  if type(value) == "table" and value.oper ~= nil then
    local index = value.index or 0
    if type(index) ~= "number" or index % 1 ~= 0 or index < 0 then
      error("output index must be a non-negative integer", 3)
    end
    return value.oper, index
  end
  if type(value) == "string" then
    local name, index = value, 0
    local base, suffix = value:match("^(.-):(%d+)$")
    if base then name, index = base, tonumber(suffix) end
    if index < 0 then error("output index must be non-negative", 3) end
    return graph:operation(name), index
  end
  error("expected an operation name, 'op:index', or a graph output", 3)
end

local function dtype_name(code)
  local name = Tensor.dtype_info(tonumber(code))
  return name
end

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

function Graph:operations()
  local lib = capi.load()
  local position = ffi.new("size_t[1]", 0)
  local names = {}
  while true do
    local operation = lib.TF_GraphNextOperation(self:handle(), position)
    if operation == nil then break end
    names[#names + 1] = ffi.string(lib.TF_OperationName(operation))
  end
  return names
end

function Graph:shape(value)
  local lib = capi.load()
  local operation, index = resolve_operation(self, value)
  if index >= lib.TF_OperationNumOutputs(operation) then
    error("operation has no output at index " .. tostring(index), 3)
  end
  local output = ffi.new("TF_Output", { oper = operation, index = index })
  local status = capi.status()
  local ndims = lib.TF_GraphGetTensorNumDims(self:handle(), output, status)
  capi.check(status, "graph tensor rank")
  if ndims < 0 then return nil end
  local shape = {}
  if ndims == 0 then return shape end
  local dims = ffi.new("int64_t[?]", ndims)
  status = capi.status()
  lib.TF_GraphGetTensorShape(self:handle(), output, dims, ndims, status)
  capi.check(status, "graph tensor shape")
  for i = 1, ndims do shape[i] = tonumber(dims[i - 1]) end
  return shape
end

function Graph:operation_info(value)
  local lib = capi.load()
  local operation, index = resolve_operation(self, value)
  local name = ffi.string(lib.TF_OperationName(operation))
  local inputs = {}
  for i = 0, lib.TF_OperationNumInputs(operation) - 1 do
    local input = ffi.new("TF_Input", { oper = operation, index = i })
    local producer = lib.TF_OperationInput(input)
    local producer_name = ffi.string(lib.TF_OperationName(producer.oper))
    inputs[#inputs + 1] = {
      input = i,
      name = producer_name .. ":" .. producer.index,
      output = producer.index,
      dtype = dtype_name(lib.TF_OperationInputType(input)),
    }
  end
  local outputs = {}
  for i = 0, lib.TF_OperationNumOutputs(operation) - 1 do
    local output = ffi.new("TF_Output", { oper = operation, index = i })
    outputs[#outputs + 1] = {
      name = name .. ":" .. i,
      index = i,
      dtype = dtype_name(lib.TF_OperationOutputType(output)),
      shape = self:shape({ oper = operation, index = i }),
      consumers = lib.TF_OperationOutputNumConsumers(output),
    }
  end
  return {
    name = name,
    op_type = ffi.string(lib.TF_OperationOpType(operation)),
    index = index,
    inputs = inputs,
    outputs = outputs,
  }
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
  local input_map = options and options.input_map
  if input_map ~= nil and type(input_map) ~= "table" then
    error("GraphDef import input_map must be a table", 2)
  end
  local mappings = {}
  if input_map ~= nil then
    local resolve_output = require("tensorflow.graph_builder").resolve_output
    for source, destination in pairs(input_map) do
      if type(source) ~= "string" then
        error("GraphDef import input_map keys must be strings", 2)
      end
      local name, index = source:match("^(.-):(%d+)$")
      name = name or source
      index = index and tonumber(index) or 0
      if name == "" then
        error("GraphDef import input_map source name must not be empty", 2)
      end
      local output = resolve_output(self, destination)
      mappings[#mappings + 1] = {
        source = source,
        name = name,
        index = index,
        output = ffi.new("TF_Output", {
          oper = output.oper, index = output.index,
        }),
      }
    end
    table.sort(mappings, function(a, b) return a.source < b.source end)
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
  for _, mapping in ipairs(mappings) do
    lib.TF_ImportGraphDefOptionsAddInputMapping(import_options, mapping.name,
      mapping.index, mapping.output)
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
