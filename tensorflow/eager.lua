--
-- Experimental eager execution over TensorFlow's TFE C API.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")

local Eager = {}
local Context = {}
Context.__index = Context

local function dtype_code(dtype)
  local _, info = Tensor.dtype_info(dtype)
  return info.code
end

local function context_handle(context)
  if context._handle == nil then error("eager context is closed", 3) end
  return context._handle
end

local function wrap_handle(handle, context)
  local keepalive = context:handle()
  context._refs = context._refs + 1
  local owner = ffi.gc(handle, function(value)
    local _ = keepalive
    capi.load().TFE_DeleteTensorHandle(value)
    context:_release()
  end)
  return setmetatable({ _handle = owner, _context = context }, Eager.Tensor)
end

local ETensor = {}
ETensor.__index = ETensor
Eager.Tensor = ETensor

function ETensor:handle()
  if self._handle == nil then error("eager tensor has been released", 2) end
  return self._handle
end

function ETensor:dtype()
  local lib = capi.load()
  local code = lib.TFE_TensorHandleDataType(self:handle())
  local name
  for _, info in pairs(Tensor._dtypes()) do
    if type(info) == "table" and info.code == code then name = info.name end
  end
  return name
end

function ETensor:shape()
  local lib = capi.load()
  local status = capi.status()
  local ndim = lib.TFE_TensorHandleNumDims(self:handle(), status)
  capi.check(status, "eager tensor rank")
  local shape = {}
  for i = 0, ndim - 1 do
    status = capi.status()
    shape[#shape + 1] = tonumber(lib.TFE_TensorHandleDim(self:handle(), i,
      status))
    capi.check(status, "eager tensor shape")
  end
  return shape
end

function ETensor:tensor()
  local status = capi.status()
  local pointer = capi.load().TFE_TensorHandleResolve(self:handle(), status)
  capi.check(status, "eager tensor resolve")
  if pointer == nil then error("TFE_TensorHandleResolve failed", 2) end
  return Tensor.wrap(pointer)
end

function ETensor:totable()
  local tensor = self:tensor()
  local value = tensor:totable()
  tensor:destroy()
  return value
end

function ETensor:format()
  return string.format("tf.eager.Tensor(shape=[%s], dtype=%s)",
    table.concat(self:shape(), ", "), tostring(self:dtype()))
end
ETensor.__tostring = ETensor.format

function Context.new(options)
  options = options or {}
  local lib = capi.load()
  local opts = capi.require(lib.TFE_NewContextOptions(), "TFE_NewContextOptions")
  lib.TFE_ContextOptionsSetAsync(opts, options.async and 1 or 0)
  local status = capi.status()
  local pointer = lib.TFE_NewContext(opts, status)
  capi.check(status, "eager context")
  lib.TFE_DeleteContextOptions(opts)
  if pointer == nil then error("TFE_NewContext failed", 2) end
  local self = setmetatable({ _refs = 0, _closed = false,
    _state = { deleted = false } }, Context)
  local handle = ffi.gc(pointer, function(value)
    if not self._state.deleted then
      self._state.deleted = true
      capi.load().TFE_DeleteContext(value)
    end
  end)
  self._handle = handle
  return self
end

function Context:handle()
  if self._closed then error("eager context is closed", 2) end
  return context_handle(self)
end

function Context:_destroy()
  if self._handle ~= nil and not self._state.deleted then
    local handle = self._handle
    self._handle = nil
    ffi.gc(handle, nil)
    self._state.deleted = true
    capi.load().TFE_DeleteContext(handle)
  end
end

function Context:_release()
  self._refs = self._refs - 1
  if self._closed and self._refs <= 0 then self:_destroy() end
end

function Context:constant(value, dtype)
  local tensor = Tensor.new(value, { dtype = dtype })
  local status = capi.status()
  local handle = capi.load().TFE_NewTensorHandle(tensor:handle(), status)
  capi.check(status, "eager tensor handle")
  if handle == nil then error("TFE_NewTensorHandle failed", 2) end
  local wrapped = wrap_handle(handle, self)
  wrapped._source = tensor
  return wrapped
end

function Context:execute(op_type, inputs, options)
  options = options or {}
  local lib = capi.load()
  local status = capi.status()
  local op = lib.TFE_NewOp(self:handle(), op_type, status)
  capi.check(status, "eager op: " .. op_type)
  if op == nil then error("TFE_NewOp failed", 2) end
  local ok, err = pcall(function()
    local input_status = capi.status()
    for _, input in ipairs(inputs or {}) do
      lib.TFE_OpAddInput(op, input:handle(), input_status)
    end
    capi.check(input_status, "eager op input")
    for name, value in pairs(options.attrs or {}) do
      if type(value) == "boolean" then
        lib.TFE_OpSetAttrBool(op, name, value and 1 or 0)
      elseif type(value) == "number" then
        lib.TFE_OpSetAttrFloat(op, name, value)
      elseif type(value) == "string" then
        local _, info = Tensor.dtype_info(value)
        lib.TFE_OpSetAttrType(op, name, info.code)
      elseif type(value) == "table" then
        local dims = ffi.new("int64_t[?]", #value)
        for i, dimension in ipairs(value) do dims[i - 1] = dimension end
        local shape_status = capi.status()
        lib.TFE_OpSetAttrShape(op, name, dims, #value, shape_status)
        capi.check(shape_status, "eager shape attribute")
      else
        error("unsupported eager attribute " .. name, 2)
      end
    end
    local outputs = ffi.new("TFE_TensorHandle*[8]")
    local count = ffi.new("int[1]", 8)
    local execute_status = capi.status()
    lib.TFE_Execute(op, outputs, count, execute_status)
    capi.check(execute_status, "eager execute: " .. op_type)
    local result = {}
    for i = 1, count[0] do result[i] = wrap_handle(outputs[i - 1], self) end
    return result
  end)
  lib.TFE_DeleteOp(op)
  if not ok then error(err, 2) end
  return err
end

local function unary(context, op_type, input, name)
  return context:execute(op_type, { input }, { attrs = { T = input:dtype() } })[1]
end

function Context:add(a, b)
  return self:execute("AddV2", { a, b }, { attrs = { T = a:dtype() } })[1]
end

function Context:sub(a, b)
  return self:execute("Sub", { a, b }, { attrs = { T = a:dtype() } })[1]
end

function Context:mul(a, b)
  return self:execute("Mul", { a, b }, { attrs = { T = a:dtype() } })[1]
end

function Context:matmul(a, b)
  return self:execute("MatMul", { a, b },
    { attrs = { T = a:dtype(), transpose_a = false, transpose_b = false } })[1]
end

function Context:relu(a)
  return unary(self, "Relu", a)
end

function Context:sigmoid(a)
  return unary(self, "Sigmoid", a)
end

function Context:square(a)
  return unary(self, "Square", a)
end

function Context:close()
  if self._closed then return end
  self._closed = true
  if self._refs <= 0 then self:_destroy() end
end

Eager.Context = Context
return Eager
