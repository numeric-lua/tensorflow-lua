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

function Graph:set_shape(value, shape)
  if shape ~= nil and type(shape) ~= "table" then
    error("graph shape must be a table or nil", 2)
  end
  if shape ~= nil then
    for i, dimension in ipairs(shape) do
      if type(dimension) ~= "number" or dimension % 1 ~= 0 or dimension < -1 then
        error("graph shape dimensions must be integers greater than or equal " ..
          "to -1 (dimension " .. tostring(i) .. ")", 2)
      end
    end
  end
  local lib = capi.load()
  local operation, index = resolve_operation(self, value)
  if index >= lib.TF_OperationNumOutputs(operation) then
    error("operation has no output at index " .. tostring(index), 2)
  end
  local output = ffi.new("TF_Output", { oper = operation, index = index })
  local dims, num_dims = nil, -1
  if shape ~= nil then
    num_dims = #shape
    dims = ffi.new("int64_t[?]", math.max(num_dims, 1))
    for i = 1, num_dims do dims[i - 1] = shape[i] end
  end
  local status = capi.status()
  lib.TF_GraphSetTensorShape(self:handle(), output, dims, num_dims, status)
  capi.check(status, "graph tensor shape update")
  return self
end

function Graph:consumers(value)
  local lib = capi.load()
  local operation, index = resolve_operation(self, value)
  if index >= lib.TF_OperationNumOutputs(operation) then
    error("operation has no output at index " .. tostring(index), 3)
  end
  local output = ffi.new("TF_Output", { oper = operation, index = index })
  local count = lib.TF_OperationOutputNumConsumers(output)
  local consumers = {}
  if count > 0 then
    local array = ffi.new("TF_Input[?]", count)
    local actual = lib.TF_OperationOutputConsumers(output, array, count)
    for i = 1, actual do
      consumers[i] = {
        name = ffi.string(lib.TF_OperationName(array[i - 1].oper)),
        input = array[i - 1].index,
      }
    end
  end
  return consumers
end

local function control_operations(lib, operation, count, getter)
  local names = {}
  if count > 0 then
    local array = ffi.new("TF_Operation*[?]", count)
    local actual = getter(operation, array, count)
    for i = 1, actual do
      names[i] = ffi.string(lib.TF_OperationName(array[i - 1]))
    end
  end
  return names
end

function Graph:control_inputs(value)
  local lib = capi.load()
  local operation = resolve_operation(self, value)
  return control_operations(lib, operation,
    lib.TF_OperationNumControlInputs(operation), lib.TF_OperationGetControlInputs)
end

function Graph:control_outputs(value)
  local lib = capi.load()
  local operation = resolve_operation(self, value)
  return control_operations(lib, operation,
    lib.TF_OperationNumControlOutputs(operation),
    lib.TF_OperationGetControlOutputs)
end

local attr_kinds = {
  [0] = "string", [1] = "int", [2] = "float", [3] = "bool", [4] = "type",
  [5] = "shape", [6] = "tensor", [7] = "placeholder", [8] = "func",
}

local function attribute_metadata(graph, value, name)
  if type(name) ~= "string" or name == "" then
    error("attribute name must be a non-empty string", 3)
  end
  local lib = capi.load()
  local operation = resolve_operation(graph, value)
  local status = capi.status()
  local metadata = lib.TF_OperationGetAttrMetadata(operation, name, status)
  capi.check(status, "graph attribute metadata")
  local kind = attr_kinds[tonumber(metadata.type)]
  if kind == nil then
    error("unknown attribute type " .. tostring(metadata.type), 3)
  end
  return metadata, kind, operation
end

local function attr_string(lib, operation, name, metadata)
  local length = tonumber(metadata.total_size)
  local capacity = math.max(length + 1, 1)
  local buffer = ffi.new("char[?]", capacity)
  local status = capi.status()
  lib.TF_OperationGetAttrString(operation, name, buffer, capacity, status)
  capi.check(status, "graph string attribute")
  return ffi.string(buffer, length)
end

local function attr_string_list(lib, operation, name, metadata)
  local count = tonumber(metadata.list_size)
  local total = math.max(tonumber(metadata.total_size), 0)
  local values = ffi.new("void*[?]", math.max(count, 1))
  local lengths = ffi.new("size_t[?]", math.max(count, 1))
  local storage = ffi.new("char[?]", math.max(total, 1))
  local status = capi.status()
  lib.TF_OperationGetAttrStringList(operation, name, values, lengths, count,
    storage, total, status)
  capi.check(status, "graph string-list attribute")
  local result = {}
  for i = 1, count do
    result[i] = ffi.string(values[i - 1], tonumber(lengths[i - 1]))
  end
  return result
end

local function attr_int_list(lib, operation, name, metadata)
  local count = tonumber(metadata.list_size)
  local values = ffi.new("int64_t[?]", math.max(count, 1))
  local status = capi.status()
  lib.TF_OperationGetAttrIntList(operation, name, values, count, status)
  capi.check(status, "graph int-list attribute")
  local result = {}
  for i = 1, count do result[i] = tonumber(values[i - 1]) end
  return result
end

local function attr_float_list(lib, operation, name, metadata)
  local count = tonumber(metadata.list_size)
  local values = ffi.new("float[?]", math.max(count, 1))
  local status = capi.status()
  lib.TF_OperationGetAttrFloatList(operation, name, values, count, status)
  capi.check(status, "graph float-list attribute")
  local result = {}
  for i = 1, count do result[i] = tonumber(values[i - 1]) end
  return result
end

local function attr_bool_list(lib, operation, name, metadata)
  local count = tonumber(metadata.list_size)
  local values = ffi.new("unsigned char[?]", math.max(count, 1))
  local status = capi.status()
  lib.TF_OperationGetAttrBoolList(operation, name, values, count, status)
  capi.check(status, "graph bool-list attribute")
  local result = {}
  for i = 1, count do result[i] = values[i - 1] ~= 0 end
  return result
end

local function attr_type_list(lib, operation, name, metadata)
  local count = tonumber(metadata.list_size)
  local values = ffi.new("TF_DataType[?]", math.max(count, 1))
  local status = capi.status()
  lib.TF_OperationGetAttrTypeList(operation, name, values, count, status)
  capi.check(status, "graph type-list attribute")
  local result = {}
  for i = 1, count do result[i] = dtype_name(values[i - 1]) end
  return result
end

local function attr_shape(lib, operation, name, metadata)
  local count = tonumber(metadata.total_size)
  if count < 0 then return nil end
  if count == 0 then return {} end
  local dims = ffi.new("int64_t[?]", count)
  local status = capi.status()
  lib.TF_OperationGetAttrShape(operation, name, dims, count, status)
  capi.check(status, "graph shape attribute")
  local result = {}
  for i = 1, count do result[i] = tonumber(dims[i - 1]) end
  return result
end

local function attr_tensor(lib, operation, name)
  local holder = ffi.new("TF_Tensor*[1]")
  local status = capi.status()
  lib.TF_OperationGetAttrTensor(operation, name, holder, status)
  capi.check(status, "graph tensor attribute")
  if holder[0] == nil then
    error("graph tensor attribute '" .. name .. "' is null", 3)
  end
  return Tensor.wrap(holder[0])
end

local function attr_shape_list(lib, operation, name, metadata)
  local count = tonumber(metadata.list_size)
  local total = tonumber(metadata.total_size)
  if total < 0 then return nil end
  local pointer = ffi.new("int64_t*[?]", math.max(count, 1))
  local dimensions = ffi.new("int[?]", math.max(count, 1))
  local storage = ffi.new("int64_t[?]", math.max(total, 1))
  local status = capi.status()
  lib.TF_OperationGetAttrShapeList(operation, name, pointer, dimensions, count,
    storage, total, status)
  capi.check(status, "graph shape-list attribute")
  local result, offset = {}, 0
  for i = 1, count do
    local shape = {}
    for dimension = 1, tonumber(dimensions[i - 1]) do
      shape[dimension] = tonumber(storage[offset + dimension - 1])
    end
    offset = offset + tonumber(dimensions[i - 1])
    result[i] = shape
  end
  return result
end

local function list_length(graph, value, arg_name, output, label)
  if type(arg_name) ~= "string" or arg_name == "" then
    error("list length argument name must be a non-empty string", 3)
  end
  local lib = capi.load()
  local operation = resolve_operation(graph, value)
  local status = capi.status()
  local length
  if output then
    length = lib.TF_OperationOutputListLength(operation, arg_name, status)
  else
    length = lib.TF_OperationInputListLength(operation, arg_name, status)
  end
  capi.check(status, label)
  return length
end

function Graph:operation_device(value)
  local lib = capi.load()
  local operation = resolve_operation(self, value)
  local device = lib.TF_OperationDevice(operation)
  if device == nil then return nil end
  local text = ffi.string(device)
  if text == "" then return nil end
  return text
end

function Graph:all_inputs(value)
  local lib = capi.load()
  local operation = resolve_operation(self, value)
  local count = lib.TF_OperationNumInputs(operation)
  local inputs = {}
  if count > 0 then
    local array = ffi.new("TF_Output[?]", count)
    lib.TF_OperationAllInputs(operation, array, count)
    for i = 1, count do
      local output = array[i - 1]
      inputs[i] = {
        input = i - 1,
        name = ffi.string(lib.TF_OperationName(output.oper)) .. ":" ..
          tostring(output.index),
        index = output.index,
        dtype = dtype_name(lib.TF_OperationOutputType(output)),
      }
    end
  end
  return inputs
end

function Graph:input_list_length(value, arg_name)
  return list_length(self, value, arg_name, false, "graph input list length")
end

function Graph:output_list_length(value, arg_name)
  return list_length(self, value, arg_name, true, "graph output list length")
end

function Graph:attribute_names(value)
  local lib = capi.load()
  local operation = resolve_operation(self, value)
  local count = lib.TF_OperationGetNumAttrs(operation)
  local names = {}
  for i = 0, count - 1 do
    local length = lib.TF_OperationGetAttrNameLength(operation, i)
    if length < 0 then
      error("operation has no attribute at index " .. tostring(i), 2)
    end
    local buffer = ffi.new("char[?]", length + 1)
    local status = capi.status()
    lib.TF_OperationGetAttrName(operation, i, buffer, status)
    capi.check(status, "graph attribute name")
    names[#names + 1] = ffi.string(buffer, length)
  end
  return names
end

function Graph:attributes(value)
  local attributes = {}
  for _, name in ipairs(self:attribute_names(value)) do
    local info = self:attribute_info(value, name)
    local undecoded = info.kind == "placeholder" or info.kind == "func" or
      (info.kind == "tensor" and info.is_list)
    if not undecoded then
      info.value = self:attribute(value, name)
    end
    attributes[#attributes + 1] = info
  end
  return attributes
end

function Graph:attribute_info(value, name)
  local metadata, kind, operation = attribute_metadata(self, value, name)
  return {
    name = name,
    operation = ffi.string(capi.load().TF_OperationName(operation)),
    kind = kind,
    is_list = metadata.is_list ~= 0,
    list_size = tonumber(metadata.list_size),
    total_size = tonumber(metadata.total_size),
  }
end

function Graph:attribute(value, name)
  local lib = capi.load()
  local metadata, kind, operation = attribute_metadata(self, value, name)
  local is_list = metadata.is_list ~= 0
  if is_list then
    if kind == "string" then
      return attr_string_list(lib, operation, name, metadata)
    elseif kind == "int" then
      return attr_int_list(lib, operation, name, metadata)
    elseif kind == "float" then
      return attr_float_list(lib, operation, name, metadata)
    elseif kind == "bool" then
      return attr_bool_list(lib, operation, name, metadata)
    elseif kind == "type" then
      return attr_type_list(lib, operation, name, metadata)
    elseif kind == "shape" then
      return attr_shape_list(lib, operation, name, metadata)
    end
  else
    if kind == "string" then
      return attr_string(lib, operation, name, metadata)
    elseif kind == "int" then
      local result = ffi.new("int64_t[1]")
      local status = capi.status()
      lib.TF_OperationGetAttrInt(operation, name, result, status)
      capi.check(status, "graph int attribute")
      return tonumber(result[0])
    elseif kind == "float" then
      local result = ffi.new("float[1]")
      local status = capi.status()
      lib.TF_OperationGetAttrFloat(operation, name, result, status)
      capi.check(status, "graph float attribute")
      return tonumber(result[0])
    elseif kind == "bool" then
      local result = ffi.new("unsigned char[1]")
      local status = capi.status()
      lib.TF_OperationGetAttrBool(operation, name, result, status)
      capi.check(status, "graph bool attribute")
      return result[0] ~= 0
    elseif kind == "type" then
      local result = ffi.new("TF_DataType[1]")
      local status = capi.status()
      lib.TF_OperationGetAttrType(operation, name, result, status)
      capi.check(status, "graph type attribute")
      return dtype_name(result[0])
    elseif kind == "shape" then
      return attr_shape(lib, operation, name, metadata)
    elseif kind == "tensor" then
      return attr_tensor(lib, operation, name)
    end
  end
  error("unsupported graph " .. (is_list and "list " or "") .. kind ..
    " attribute '" .. name .. "'", 2)
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
      consumer_ops = self:consumers({ oper = operation, index = i }),
    }
  end
  return {
    name = name,
    op_type = ffi.string(lib.TF_OperationOpType(operation)),
    index = index,
    inputs = inputs,
    outputs = outputs,
    control_inputs = self:control_inputs({ oper = operation, index = index }),
    control_outputs = self:control_outputs({ oper = operation, index = index }),
  }
end

function Graph.output_spec(_, output)
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
  local return_elements = options and options.return_elements
  if return_elements ~= nil then
    if type(return_elements) ~= "table" then
      error("GraphDef import return_elements must be a table", 2)
    end
    for i, spec in ipairs(return_elements) do
      if type(spec) ~= "string" then
        error("GraphDef import return_elements[" .. i ..
          "] must be a string", 2)
      end
    end
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
  if return_elements ~= nil then
    local outputs = {}
    for i, spec in ipairs(return_elements) do
      local operation, index = resolve_operation(self, spec)
      if index >= lib.TF_OperationNumOutputs(operation) then
        error("operation has no output at index " .. tostring(index), 2)
      end
      local base = spec:match("^(.-):(%d+)$")
      local name = (base or spec) .. ":" .. tostring(index)
      local output = ffi.new("TF_Output", { oper = operation, index = index })
      outputs[i] = {
        oper = operation,
        index = index,
        name = name,
        dtype = dtype_name(lib.TF_OperationOutputType(output)),
        shape = self:shape({ oper = operation, index = index }),
      }
    end
    return outputs
  end
  return self
end

function Graph:import_file(path, options)
  local file, err = io.open(path, "rb")
  if file == nil then error("cannot open GraphDef '" .. path .. "': " .. tostring(err), 2) end
  local data = file:read("*a")
  file:close()
  return self:import_graph_def(data, options)
end

function Graph:node_def(value)
  local lib = capi.load()
  local operation = resolve_operation(self, value)
  local buffer = ffi.gc(capi.require(lib.TF_NewBuffer(), "TF_NewBuffer"),
    function(pointer) lib.TF_DeleteBuffer(pointer) end)
  local status = capi.status()
  lib.TF_OperationToNodeDef(operation, buffer, status)
  capi.check(status, "NodeDef export")
  local struct = lib.TF_GetBuffer(buffer)
  if struct.length > 0 and struct.data == nil then
    error("TF_OperationToNodeDef returned no data", 2)
  end
  local data = struct.length > 0 and ffi.string(struct.data, struct.length) or ""
  ffi.gc(buffer, nil)
  lib.TF_DeleteBuffer(buffer)
  return data
end

function Graph:op_def(op_type)
  if type(op_type) ~= "string" or op_type == "" then
    error("op type must be a non-empty string", 2)
  end
  local lib = capi.load()
  local buffer = ffi.gc(capi.require(lib.TF_NewBuffer(), "TF_NewBuffer"),
    function(pointer) lib.TF_DeleteBuffer(pointer) end)
  local status = capi.status()
  lib.TF_GraphGetOpDef(self:handle(), op_type, buffer, status)
  capi.check(status, "OpDef export")
  local struct = lib.TF_GetBuffer(buffer)
  if struct.length > 0 and struct.data == nil then
    error("TF_GraphGetOpDef returned no data", 2)
  end
  local data = struct.length > 0 and ffi.string(struct.data, struct.length) or ""
  ffi.gc(buffer, nil)
  lib.TF_DeleteBuffer(buffer)
  return data
end

function Graph:version_def()
  local lib = capi.load()
  local buffer = ffi.gc(capi.require(lib.TF_NewBuffer(), "TF_NewBuffer"),
    function(pointer) lib.TF_DeleteBuffer(pointer) end)
  local status = capi.status()
  lib.TF_GraphVersions(self:handle(), buffer, status)
  capi.check(status, "graph versions export")
  local struct = lib.TF_GetBuffer(buffer)
  if struct.length > 0 and struct.data == nil then
    error("TF_GraphVersions returned no data", 2)
  end
  local data = struct.length > 0 and ffi.string(struct.data, struct.length) or ""
  ffi.gc(buffer, nil)
  lib.TF_DeleteBuffer(buffer)
  return data
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
