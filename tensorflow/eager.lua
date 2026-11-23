--
-- Experimental eager execution over TensorFlow's TFE C API.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")

local Eager = {}
local Context = {}
Context.__index = Context

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
    capi.released.eager_tensors = capi.released.eager_tensors + 1
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

function ETensor:destroy()
  if self._handle == nil then return end
  local handle = self._handle
  local context = self._context
  self._handle = nil
  self._source = nil
  ffi.gc(handle, nil)
  capi.load().TFE_DeleteTensorHandle(handle)
  capi.released.eager_tensors = capi.released.eager_tensors + 1
  if context ~= nil then context:_release() end
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
  local ok, err = pcall(capi.check, status, "eager context")
  lib.TFE_DeleteContextOptions(opts)
  if not ok or pointer == nil then
    error(err or "TFE_NewContext failed", 2)
  end
  local state = { deleted = false }
  local handle = ffi.gc(pointer, function(value)
    if not state.deleted then
      state.deleted = true
      capi.load().TFE_DeleteContext(value)
      capi.released.contexts = capi.released.contexts + 1
    end
  end)
  return setmetatable({ _refs = 0, _closed = false, _state = state,
    _handle = handle }, Context)
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
    capi.released.contexts = capi.released.contexts + 1
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
        if value.int ~= nil then
          lib.TFE_OpSetAttrInt(op, name, value.int)
        elseif value.float ~= nil then
          lib.TFE_OpSetAttrFloat(op, name, value.float)
        elseif value.bool ~= nil then
          lib.TFE_OpSetAttrBool(op, name, value.bool and 1 or 0)
        elseif value.type ~= nil then
          local _, info = Tensor.dtype_info(value.type)
          lib.TFE_OpSetAttrType(op, name, info.code)
        elseif value.ints ~= nil then
          local ints = ffi.new("int64_t[?]", #value.ints)
          for i, integer in ipairs(value.ints) do ints[i - 1] = integer end
          lib.TFE_OpSetAttrIntList(op, name, ints, #value.ints)
        elseif value.shape ~= nil then
          local dims = ffi.new("int64_t[?]", #value.shape)
          for i, dimension in ipairs(value.shape) do dims[i - 1] = dimension end
          local shape_status = capi.status()
          lib.TFE_OpSetAttrShape(op, name, dims, #value.shape, shape_status)
          capi.check(shape_status, "eager shape attribute")
        else
          error("unsupported eager attribute table " .. name, 2)
        end
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

local function unary(context, op_type, input)
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

function Context:div(a, b)
  return self:execute("RealDiv", { a, b }, { attrs = { T = a:dtype() } })[1]
end

function Context:cast(a, dst_dtype)
  local dtype = Tensor.dtype_info(dst_dtype)
  return self:execute("Cast", { a },
    { attrs = { SrcT = a:dtype(), DstT = dtype } })[1]
end

function Context:transpose(a, perm)
  if type(perm) ~= "table" or #perm == 0 then
    error("eager transpose expects a non-empty permutation table", 2)
  end
  local permutation = self:constant(perm, "int32")
  return self:execute("Transpose", { a, permutation },
    { attrs = { T = a:dtype(), Tperm = "int32" } })[1]
end

function Context:expand_dims(a, axis)
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("eager expand_dims expects an integer axis", 2)
  end
  local dimension = self:constant(axis, "int32")
  return self:execute("ExpandDims", { a, dimension },
    { attrs = { T = a:dtype(), Tdim = "int32" } })[1]
end

function Context:squeeze(a, axis)
  local attrs = { T = a:dtype() }
  if axis ~= nil then
    if type(axis) == "number" then axis = { axis } end
    if type(axis) ~= "table" then
      error("eager squeeze expects an integer axis or table", 2)
    end
    attrs.squeeze_dims = { ints = axis }
  end
  return self:execute("Squeeze", { a }, { attrs = attrs })[1]
end

local function reduction(context, op_type, input, axes, keep_dims)
  if type(axes) ~= "table" then
    error("eager " .. op_type:lower() .. " expects an axes table", 3)
  end
  for _, axis in ipairs(axes) do
    if type(axis) ~= "number" or axis % 1 ~= 0 then
      error("eager reduction axes must be integers", 3)
    end
  end
  local axis_tensor = context:constant(axes, "int32")
  return context:execute(op_type, { input, axis_tensor },
    { attrs = { T = input:dtype(), Tidx = "int32",
      keep_dims = keep_dims and true or false } })[1]
end

function Context:reduce_sum(a, axes, keep_dims)
  return reduction(self, "Sum", a, axes, keep_dims)
end

function Context:reduce_mean(a, axes, keep_dims)
  return reduction(self, "Mean", a, axes, keep_dims)
end

function Context:reduce_max(a, axes, keep_dims)
  return reduction(self, "Max", a, axes, keep_dims)
end

function Context:reduce_min(a, axes, keep_dims)
  return reduction(self, "Min", a, axes, keep_dims)
end

function Context:reduce_prod(a, axes, keep_dims)
  return reduction(self, "Prod", a, axes, keep_dims)
end

function Context:argmax(a, axis, output_dtype)
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("eager argmax expects an integer axis", 2)
  end
  local dtype = Tensor.dtype_info(output_dtype or "int64")
  local dimension = self:constant(axis, "int32")
  return self:execute("ArgMax", { a, dimension },
    { attrs = { T = a:dtype(), Tidx = "int32", output_type = dtype } })[1]
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

function Context:exp(a)
  return unary(self, "Exp", a)
end

function Context:log(a)
  return unary(self, "Log", a)
end

function Context:sqrt(a)
  return unary(self, "Sqrt", a)
end

function Context:abs(a)
  return unary(self, "Abs", a)
end

function Context:tanh(a)
  return unary(self, "Tanh", a)
end

function Context:softplus(a)
  return unary(self, "Softplus", a)
end

function Context:softmax(a)
  return unary(self, "Softmax", a)
end

function Context:elu(a)
  return unary(self, "Elu", a)
end

function Context:leaky_relu(a, alpha)
  alpha = alpha or 0.2
  if type(alpha) ~= "number" then
    error("eager leaky_relu alpha must be a number", 2)
  end
  return self:execute("LeakyRelu", { a },
    { attrs = { T = a:dtype(), alpha = alpha } })[1]
end

function Context:reshape(a, shape)
  if type(shape) ~= "table" then error("eager reshape expects a shape table", 2) end
  local shape_tensor = self:constant(shape, "int32")
  return self:execute("Reshape", { a, shape_tensor },
    { attrs = { T = a:dtype(), Tshape = "int32" } })[1]
end

function Context:close()
  if self._closed then return end
  self._closed = true
  if self._refs <= 0 then self:_destroy() end
end

Eager.Context = Context
return Eager
