--
-- Lua tensors backed by TF_Tensor.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")

local Tensor = {}
Tensor.__index = Tensor

local dtypes = {
  float32 = { code = 1, ctype = "float", size = 4, name = "float32" },
  ["float"] = "float32",
  float64 = { code = 2, ctype = "double", size = 8, name = "float64" },
  ["double"] = "float64",
  int32 = { code = 3, ctype = "int32_t", size = 4, name = "int32" },
  ["int"] = "int32",
  uint8 = { code = 4, ctype = "uint8_t", size = 1, name = "uint8" },
  int16 = { code = 5, ctype = "int16_t", size = 2, name = "int16" },
  int8 = { code = 6, ctype = "int8_t", size = 1, name = "int8" },
  int64 = { code = 9, ctype = "int64_t", size = 8, name = "int64" },
  ["long"] = "int64",
  bool = { code = 10, ctype = "uint8_t", size = 1, name = "bool" },
  uint16 = { code = 17, ctype = "uint16_t", size = 2, name = "uint16" },
  uint32 = { code = 22, ctype = "uint32_t", size = 4, name = "uint32" },
  uint64 = { code = 23, ctype = "uint64_t", size = 8, name = "uint64" },
}

local function canonical(dtype)
  if dtype == nil then return "float32" end
  if type(dtype) == "number" then
    for name, info in pairs(dtypes) do
      if type(info) == "table" and info.code == dtype then return name end
    end
  end
  if type(dtype) ~= "string" then
    error("dtype must be a string or TF_DataType id", 3)
  end
  dtype = dtype:lower()
  local info = dtypes[dtype]
  if type(info) == "string" then dtype = info; info = dtypes[dtype] end
  if not info then error("unsupported TensorFlow dtype: " .. tostring(dtype), 3) end
  return dtype
end

local function dtype_info(dtype, level)
  dtype = canonical(dtype)
  local info = dtypes[dtype]
  if not info then error("unsupported TensorFlow dtype: " .. tostring(dtype), level or 3) end
  return dtype, info
end

local function shape_of(value)
  if type(value) ~= "table" then return {} end
  local n = #value
  if n == 0 then return { 0 } end
  local child
  for i = 1, n do
    if type(value[i]) ~= type(value[1]) then
      error("tensor table must be rectangular", 4)
    end
    local current = shape_of(value[i])
    if child == nil then child = current
    elseif #child ~= #current then
      error("tensor table must be rectangular", 4)
    else
      for d = 1, #child do
        if child[d] ~= current[d] then
          error("tensor table must be rectangular", 4)
        end
      end
    end
  end
  local shape = { n }
  for _, dim in ipairs(child or {}) do shape[#shape + 1] = dim end
  return shape
end

local function flatten(value, out)
  if type(value) ~= "table" then out[#out + 1] = value; return end
  for i = 1, #value do flatten(value[i], out) end
end

local function make_dims(shape)
  if #shape == 0 then return nil, 0 end
  local dims = ffi.new("int64_t[?]", #shape)
  for i = 1, #shape do dims[i - 1] = shape[i] end
  return dims, #shape
end

local function product(shape)
  local n = 1
  for _, d in ipairs(shape) do n = n * d end
  return n
end

local function scalar(value, info)
  if info.name == "bool" then return value and 1 or 0 end
  return value
end

function Tensor.new(value, options)
  options = options or {}
  if type(options) ~= "table" then
    error("tensor options must be a table or nil", 3)
  end
  if getmetatable(value) == Tensor then return value end
  local dtype, info = dtype_info(options.dtype)
  local shape = shape_of(value)
  local values = {}
  flatten(value, values)
  local count = product(shape)
  if #values ~= count then
    error(string.format("tensor data has %d values but shape needs %d",
      #values, count), 2)
  end
  local lib = capi.load()
  local dims, ndim = make_dims(shape)
  local ptr = lib.TF_AllocateTensor(info.code, dims, ndim,
    count * info.size)
  if ptr == nil then error("TF_AllocateTensor failed", 2) end
  local data = lib.TF_TensorData(ptr)
  if count > 0 then
    local array = ffi.cast(info.ctype .. "*", data)
    for i = 1, count do array[i - 1] = scalar(values[i], info) end
  end
  return setmetatable({ _ptr = ptr, _owned = true, _shape = shape,
    _dtype = dtype }, Tensor)
end

function Tensor.wrap(ptr)
  if ptr == nil then return nil end
  local lib = capi.load()
  local ndim = lib.TF_NumDims(ptr)
  local shape = {}
  for i = 0, ndim - 1 do shape[#shape + 1] = tonumber(lib.TF_Dim(ptr, i)) end
  local code = lib.TF_TensorType(ptr)
  local name
  for candidate, info in pairs(dtypes) do
    if type(info) == "table" and info.code == code then name = info.name end
  end
  if name == nil then
    lib.TF_DeleteTensor(ptr)
    error("unsupported output TensorFlow dtype id: " .. tostring(code), 2)
  end
  return setmetatable({ _ptr = ptr, _owned = true, _shape = shape,
    _dtype = name }, Tensor)
end

function Tensor:handle()
  return self._ptr
end

function Tensor:shape()
  local copy = {}
  for i, d in ipairs(self._shape) do copy[i] = d end
  return copy
end
Tensor.sizes = Tensor.shape

function Tensor:dtype()
  return self._dtype
end

function Tensor:dim()
  return #self._shape
end

function Tensor:numel()
  return product(self._shape)
end

function Tensor:nbytes()
  return tonumber(capi.load().TF_TensorByteSize(self._ptr))
end

function Tensor:totable()
  local _, info = dtype_info(self._dtype)
  local lib = capi.load()
  local count = self:numel()
  local flat = {}
  if count > 0 then
    local array = ffi.cast(info.ctype .. "*", lib.TF_TensorData(self._ptr))
    for i = 1, count do
      local value = array[i - 1]
      if info.name == "bool" then value = value ~= 0 end
      flat[i] = value
    end
  end
  local position = 1
  local function build(dim)
    if dim > #self._shape then
      local value = flat[position]
      position = position + 1
      return value
    end
    local result = {}
    for i = 1, self._shape[dim] do result[i] = build(dim + 1) end
    return result
  end
  if #self._shape == 0 then return flat[1] end
  return build(1)
end

function Tensor:format()
  return string.format("tf.Tensor(shape=[%s], dtype=%s)",
    table.concat(self._shape, ", "), self._dtype)
end

Tensor.__tostring = Tensor.format

Tensor.__gc = function(self)
  if self._owned and self._ptr ~= nil then
    capi.load().TF_DeleteTensor(self._ptr)
    self._ptr = nil
  end
end

function Tensor.dtype_info(dtype)
  return dtype_info(dtype)
end

return Tensor
