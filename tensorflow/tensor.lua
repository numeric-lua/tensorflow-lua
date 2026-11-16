--
-- Lua tensors backed by TF_Tensor.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")

local Tensor = {}
Tensor.__index = Tensor

local Complex = {}
Complex.__index = Complex

local function is_complex(value)
  return type(value) == "table" and getmetatable(value) == Complex
end

function Tensor.complex(real, imag)
  if type(real) ~= "number" or type(imag) ~= "number" then
    error("tf.complex expects real and imaginary numbers", 2)
  end
  return setmetatable({ real = real, imag = imag }, Complex)
end

function Complex:format()
  return string.format("(%g%+gi)", self.real, self.imag)
end
Complex.__tostring = Complex.format

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
  qint8 = { code = 11, ctype = "int8_t", size = 1, name = "qint8" },
  quint8 = { code = 12, ctype = "uint8_t", size = 1, name = "quint8" },
  qint32 = { code = 13, ctype = "int32_t", size = 4, name = "qint32" },
  qint16 = { code = 15, ctype = "int16_t", size = 2, name = "qint16" },
  quint16 = { code = 16, ctype = "uint16_t", size = 2, name = "quint16" },
  uint16 = { code = 17, ctype = "uint16_t", size = 2, name = "uint16" },
  uint32 = { code = 22, ctype = "uint32_t", size = 4, name = "uint32" },
  uint64 = { code = 23, ctype = "uint64_t", size = 8, name = "uint64" },
  string = { code = 7, ctype = nil, size = 24, name = "string" },
  complex64 = { code = 8, ctype = "float", size = 8, name = "complex64",
    complex = true },
  complex128 = { code = 18, ctype = "double", size = 16, name = "complex128",
    complex = true },
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
  if is_complex(value) or type(value) ~= "table" then return {} end
  local count = 0
  for key in pairs(value) do
    if type(key) ~= "number" or key % 1 ~= 0 or key < 1 then
      error("tensor table must be a contiguous array, found non-array key " ..
        tostring(key), 4)
    end
    count = count + 1
  end
  for i = 1, count do
    if value[i] == nil then
      error("tensor table must not contain nil holes", 4)
    end
  end
  if count == 0 then return { 0 } end
  local child
  for i = 1, count do
    if (type(value[i]) == "table") ~= (type(value[1]) == "table") then
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
  local shape = { count }
  for _, dim in ipairs(child or {}) do shape[#shape + 1] = dim end
  return shape
end

local function flatten(value, out)
  if is_complex(value) or type(value) ~= "table" then
    out[#out + 1] = value
    return
  end
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
  if info.complex then
    if type(value) == "number" then return value, 0 end
    if is_complex(value) then return value.real, value.imag end
    error("complex tensors require tf.complex or numbers", 4)
  end
  if info.name == "string" then
    if type(value) ~= "string" then
      error("string tensors require strings, got " .. type(value), 4)
    end
    return value
  end
  if info.name == "bool" then
    if value == false or value == nil or value == 0 then return 0 end
    if value == true or value == 1 then return 1 end
    error("bool tensors accept booleans or 0/1, got " .. tostring(value), 4)
  end
  if type(value) ~= "number" then
    error("numeric tensors require numbers, got " .. type(value), 4)
  end
  return value
end

function Tensor.new(value, options)
  options = options or {}
  if type(options) ~= "table" then
    error("tensor options must be a table or nil", 3)
  end
  if getmetatable(value) == Tensor then
    if options.dtype and canonical(options.dtype) ~= value:dtype() then
      error("tf.tensor cannot convert an existing Tensor; use a named cast",
        3)
    end
    return value
  end
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
  if info.complex then
    local ptr = lib.TF_AllocateTensor(info.code, dims, ndim,
      count * info.size)
    if ptr == nil then error("TF_AllocateTensor failed", 2) end
    local ok, err = pcall(function()
      local data = lib.TF_TensorData(ptr)
      if count > 0 and data == nil then
        error("TF_TensorData returned NULL for a non-empty tensor", 2)
      end
      local array = ffi.cast(info.ctype .. "*", data)
      for i = 1, count do
        local real, imaginary = scalar(values[i], info)
        array[(i - 1) * 2] = real
        array[(i - 1) * 2 + 1] = imaginary
      end
    end)
    if not ok then lib.TF_DeleteTensor(ptr); error(err, 2) end
    local handle = ffi.gc(ptr, function(pointer)
      capi.release_tensor(pointer)
    end)
    return setmetatable({ _handle = handle, _shape = shape,
      _dtype = dtype }, Tensor)
  end
  if info.name == "string" then
    local ptr = lib.TF_AllocateTensor(info.code, dims, ndim,
      count * info.size)
    if ptr == nil then error("TF_AllocateTensor failed", 2) end
    local ok, err = pcall(function()
      local data = lib.TF_TensorData(ptr)
      if count > 0 and data == nil then
        error("TF_TensorData returned NULL for a non-empty tensor", 2)
      end
      local array = ffi.cast("TF_TString*", data)
      for i = 1, count do
        local element = values[i]
        if type(element) ~= "string" then
          error("string tensors require strings, got " .. type(element), 2)
        end
        lib.TF_StringInit(array + (i - 1))
        lib.TF_StringCopy(array + (i - 1), element, #element)
      end
    end)
    if not ok then lib.TF_DeleteTensor(ptr); error(err, 2) end
    local handle = ffi.gc(ptr, function(pointer)
      capi.release_tensor(pointer)
    end)
    return setmetatable({ _handle = handle, _shape = shape,
      _dtype = dtype }, Tensor)
  end
  local ptr = lib.TF_AllocateTensor(info.code, dims, ndim,
    count * info.size)
  if ptr == nil then error("TF_AllocateTensor failed", 2) end
  local ok, err = pcall(function()
    local data = lib.TF_TensorData(ptr)
    if count > 0 and data == nil then
      error("TF_TensorData returned NULL for a non-empty tensor", 2)
    end
    if count > 0 then
      local array = ffi.cast(info.ctype .. "*", data)
      for i = 1, count do array[i - 1] = scalar(values[i], info) end
    end
  end)
  if not ok then
    lib.TF_DeleteTensor(ptr)
    error(err, 2)
  end
  local handle = ffi.gc(ptr, function(pointer)
    capi.release_tensor(pointer)
  end)
  return setmetatable({ _handle = handle, _shape = shape,
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
  for _, info in pairs(dtypes) do
    if type(info) == "table" and info.code == code then name = info.name end
  end
  if name == nil then
    error("unsupported output TensorFlow dtype id: " .. tostring(code), 2)
  end
  local handle = ffi.gc(ptr, function(pointer)
    capi.release_tensor(pointer)
  end)
  return setmetatable({ _handle = handle, _shape = shape,
    _dtype = name }, Tensor)
end

function Tensor:handle()
  if self._handle == nil then error("tensor has been destroyed", 2) end
  return self._handle
end

function Tensor:destroy()
  if self._handle ~= nil then
    local handle = self._handle
    self._handle = nil
    ffi.gc(handle, nil)
    capi.load().TF_DeleteTensor(handle)
  end
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
  return tonumber(capi.load().TF_TensorByteSize(self:handle()))
end

function Tensor:totable()
  local _, info = dtype_info(self._dtype)
  local lib = capi.load()
  local count = self:numel()
  local flat = {}
  if count > 0 and info.complex then
    local array = ffi.cast(info.ctype .. "*", lib.TF_TensorData(self:handle()))
    for i = 1, count do
      flat[i] = Tensor.complex(array[(i - 1) * 2], array[(i - 1) * 2 + 1])
    end
  elseif count > 0 and info.name == "string" then
    local array = ffi.cast("TF_TString*", lib.TF_TensorData(self:handle()))
    for i = 1, count do
      local pointer = lib.TF_StringGetDataPointer(array + (i - 1))
      local size = tonumber(lib.TF_StringGetSize(array + (i - 1)))
      flat[i] = pointer ~= nil and ffi.string(pointer, size) or ""
    end
  elseif count > 0 then
    local array = ffi.cast(info.ctype .. "*", lib.TF_TensorData(self:handle()))
    for i = 1, count do
      local value = array[i - 1]
      if info.name == "bool" then value = value ~= 0
      elseif info.name:match("^q?u?int") then value = tonumber(value) end
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

function Tensor:bytes()
  if self._dtype == "string" then
    error("string tensors do not support raw byte export", 2)
  end
  local lib = capi.load()
  local count = self:nbytes()
  if count == 0 then return "" end
  local data = lib.TF_TensorData(self:handle())
  if data == nil then error("TF_TensorData returned NULL", 2) end
  return ffi.string(data, count)
end

function Tensor.from_bytes(dtype, shape, bytes)
  local name, info = dtype_info(dtype)
  if name == "string" then
    error("string tensors do not support from_bytes", 2)
  end
  local count = product(shape)
  local expected = count * info.size
  if #bytes ~= expected then
    error(string.format("tensor bytes have length %d, expected %d",
      #bytes, expected), 2)
  end
  local lib = capi.load()
  local dims, ndim = make_dims(shape)
  local ptr = lib.TF_AllocateTensor(info.code, dims, ndim, expected)
  if ptr == nil then error("TF_AllocateTensor failed", 2) end
  if expected > 0 then
    local data = lib.TF_TensorData(ptr)
    if data == nil then
      lib.TF_DeleteTensor(ptr)
      error("TF_TensorData returned NULL", 2)
    end
    ffi.copy(data, bytes, expected)
  end
  local handle = ffi.gc(ptr, function(pointer)
    capi.release_tensor(pointer)
  end)
  return setmetatable({ _handle = handle, _shape = shape, _dtype = name },
    Tensor)
end

function Tensor:format()
  return string.format("tf.Tensor(shape=[%s], dtype=%s)",
    table.concat(self._shape, ", "), self._dtype)
end

Tensor.__tostring = Tensor.format

function Tensor._dtypes()
  return dtypes
end

function Tensor.dtype_info(dtype)
  return dtype_info(dtype)
end

return Tensor
