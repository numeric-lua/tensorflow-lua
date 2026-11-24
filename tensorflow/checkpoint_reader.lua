--
-- Read variables from official TensorFlow checkpoints.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")

local Reader = {}
Reader.__index = Reader

function Reader.new(path)
  local lib = capi.load()
  local status = capi.status()
  local pointer = lib.TF_NewCheckpointReader(path, status)
  capi.check(status, "checkpoint open")
  if pointer == nil then error("TF_NewCheckpointReader failed", 2) end
  return setmetatable({ _handle = ffi.gc(pointer,
    function(value) capi.release_reader(value) end) }, Reader)
end

function Reader:handle()
  if self._handle == nil then error("checkpoint reader is closed", 2) end
  return self._handle
end

function Reader:size()
  return capi.load().TF_CheckpointReaderSize(self:handle())
end

function Reader:variable(index)
  local name = capi.load().TF_CheckpointReaderGetVariable(self:handle(), index)
  if name == nil then error("checkpoint variable index out of range", 2) end
  return ffi.string(name)
end

function Reader:variables()
  local result = {}
  for i = 0, self:size() - 1 do result[#result + 1] = self:variable(i) end
  return result
end

function Reader:has(name)
  return capi.load().TF_CheckpointReaderHasTensor(self:handle(), name) ~= 0
end

local function require_variable(reader, name)
  if type(name) ~= "string" or name == "" then
    error("checkpoint variable name must be a non-empty string", 3)
  end
  if not reader:has(name) then
    error("checkpoint has no variable '" .. name .. "'", 2)
  end
end

function Reader:dtype(name)
  require_variable(self, name)
  local code = capi.load().TF_CheckpointReaderGetVariableDataType(self:handle(),
    name)
  return (Tensor.dtype_info(tonumber(code)))
end

function Reader:shape(name)
  require_variable(self, name)
  local lib = capi.load()
  local ndims = lib.TF_CheckpointReaderGetVariableNumDims(self:handle(), name)
  if ndims < 0 then return nil end
  if ndims == 0 then return {} end
  local dims = ffi.new("int64_t[?]", ndims)
  local status = capi.status()
  lib.TF_CheckpointReaderGetVariableShape(self:handle(), name, dims, ndims,
    status)
  capi.check(status, "checkpoint variable shape")
  local shape = {}
  for i = 1, ndims do shape[i] = tonumber(dims[i - 1]) end
  return shape
end

function Reader:info(name)
  return { name = name, dtype = self:dtype(name), shape = self:shape(name) }
end

function Reader:tensor(name)
  local lib = capi.load()
  local status = capi.status()
  local pointer = lib.TF_CheckpointReaderGetTensor(self:handle(), name, status)
  capi.check(status, "checkpoint tensor")
  if pointer == nil then error("checkpoint has no tensor '" .. name .. "'", 2) end
  return Tensor.wrap(pointer)
end

function Reader:close()
  if self._handle ~= nil then
    local handle = self._handle
    self._handle = nil
    ffi.gc(handle, nil)
    capi.release_reader(handle)
  end
end

return Reader
