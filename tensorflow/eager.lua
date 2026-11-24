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
  local capacity = options.num_outputs or 8
  if type(capacity) ~= "number" or capacity % 1 ~= 0 or capacity < 1 then
    error("eager num_outputs must be a positive integer", 2)
  end
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
        if value.string ~= nil then
          if type(value.string) ~= "string" then
            error("eager string attribute " .. name .. " must be a string", 2)
          end
          lib.TFE_OpSetAttrString(op, name, value.string, #value.string)
        elseif value.int ~= nil then
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
    local outputs = ffi.new("TFE_TensorHandle*[?]", capacity)
    local count = ffi.new("int[1]", capacity)
    local execute_status = capi.status()
    lib.TFE_Execute(op, outputs, count, execute_status)
    capi.check(execute_status, "eager execute: " .. op_type)
    if count[0] > capacity then
      error("eager op " .. op_type .. " returned " .. tostring(count[0]) ..
        " outputs, expected at most " .. tostring(capacity), 2)
    end
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

local function comparison(context, op_type, a, b)
  return context:execute(op_type, { a, b },
    { attrs = { T = a:dtype() } })[1]
end

function Context:equal(a, b)
  return comparison(self, "Equal", a, b)
end

function Context:not_equal(a, b)
  return comparison(self, "NotEqual", a, b)
end

function Context:greater(a, b)
  return comparison(self, "Greater", a, b)
end

function Context:greater_equal(a, b)
  return comparison(self, "GreaterEqual", a, b)
end

function Context:less(a, b)
  return comparison(self, "Less", a, b)
end

function Context:less_equal(a, b)
  return comparison(self, "LessEqual", a, b)
end

local function logical_binary(context, op_type, a, b)
  return context:execute(op_type, { a, b })[1]
end

function Context:logical_and(a, b)
  return logical_binary(self, "LogicalAnd", a, b)
end

function Context:logical_or(a, b)
  return logical_binary(self, "LogicalOr", a, b)
end

function Context:logical_xor(a, b)
  return comparison(self, "NotEqual", a, b)
end

function Context:logical_not(a)
  return self:execute("LogicalNot", { a })[1]
end

function Context:maximum(a, b)
  return self:execute("Maximum", { a, b },
    { attrs = { T = a:dtype() } })[1]
end

function Context:minimum(a, b)
  return self:execute("Minimum", { a, b },
    { attrs = { T = a:dtype() } })[1]
end

function Context:matmul(a, b)
  return self:execute("MatMul", { a, b },
    { attrs = { T = a:dtype(), transpose_a = false, transpose_b = false } })[1]
end

function Context:batch_matmul(a, b, options)
  options = options or {}
  local adj_x = options.adj_x
  if adj_x == nil then adj_x = options.transpose_a end
  local adj_y = options.adj_y
  if adj_y == nil then adj_y = options.transpose_b end
  if adj_x ~= nil and type(adj_x) ~= "boolean" then
    error("eager batch_matmul adj_x must be a boolean", 2)
  end
  if adj_y ~= nil and type(adj_y) ~= "boolean" then
    error("eager batch_matmul adj_y must be a boolean", 2)
  end
  return self:execute("BatchMatMulV2", { a, b },
    { attrs = { T = a:dtype(), adj_x = adj_x == true,
      adj_y = adj_y == true } })[1]
end

function Context:fill(dims, value, dtype)
  if type(dims) ~= "table" then
    error("eager fill expects a dims table", 2)
  end
  for i, dimension in ipairs(dims) do
    if type(dimension) ~= "number" or dimension % 1 ~= 0 or dimension < 0 then
      error("eager fill dimensions must be non-negative integers (dimension " ..
        tostring(i) .. ")", 2)
    end
  end
  local dims_tensor = self:constant(dims, "int32")
  local value_tensor
  if getmetatable(value) == ETensor then
    value_tensor = value
  else
    value_tensor = self:constant(value, dtype or "float32")
  end
  return self:execute("Fill", { dims_tensor, value_tensor },
    { attrs = { T = value_tensor:dtype(), index_type = "int32" } })[1]
end

function Context:range(start, limit, delta, dtype)
  local value_dtype = dtype or "float32"
  local start_tensor = self:constant(start, value_dtype)
  local limit_tensor = self:constant(limit, value_dtype)
  local delta_tensor = self:constant(delta, value_dtype)
  return self:execute("Range", { start_tensor, limit_tensor, delta_tensor },
    { attrs = { Tidx = value_dtype } })[1]
end

function Context:linspace(start, stop, num, dtype)
  local value_dtype = dtype or "float32"
  if value_dtype ~= "float32" and value_dtype ~= "float64" then
    error("eager linspace dtype must be float32 or float64", 2)
  end
  if type(num) ~= "number" or num % 1 ~= 0 or num < 1 then
    error("eager linspace num must be a positive integer", 2)
  end
  local start_tensor = self:constant(start, value_dtype)
  local stop_tensor = self:constant(stop, value_dtype)
  local num_tensor = self:constant(num, "int32")
  return self:execute("LinSpace", { start_tensor, stop_tensor, num_tensor },
    { attrs = { T = value_dtype, Tidx = "int32" } })[1]
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

function Context:reduce_all(a, axes, keep_dims)
  return reduction(self, "All", a, axes, keep_dims)
end

function Context:reduce_any(a, axes, keep_dims)
  return reduction(self, "Any", a, axes, keep_dims)
end

function Context:reduce_variance(a, axes, keep_dims)
  if type(axes) ~= "table" then
    error("eager reduce_variance expects an axes table", 2)
  end
  local mean = self:reduce_mean(a, axes, true)
  local centered = self:sub(a, mean)
  local squared = self:mul(centered, centered)
  return self:reduce_mean(squared, axes, keep_dims)
end

function Context:reduce_stddev(a, axes, keep_dims)
  if type(axes) ~= "table" then
    error("eager reduce_stddev expects an axes table", 2)
  end
  return self:sqrt(self:reduce_variance(a, axes, keep_dims))
end

function Context:reduce_logsumexp(a, axes, keep_dims)
  if type(axes) ~= "table" then
    error("eager reduce_logsumexp expects an axes table", 2)
  end
  local maximum = self:reduce_max(a, axes, true)
  local centered = self:sub(a, maximum)
  local total = self:reduce_sum(self:exp(centered), axes, keep_dims)
  local logarithm = self:log(total)
  local reduced_max = keep_dims and maximum or self:reduce_max(a, axes, false)
  return self:add(logarithm, reduced_max)
end

function Context:add_n(inputs)
  if type(inputs) ~= "table" or #inputs == 0 then
    error("eager add_n expects a non-empty input list", 2)
  end
  return self:execute("AddN", inputs,
    { attrs = { T = inputs[1]:dtype(), N = { int = #inputs } } })[1]
end

local function arg_reduction(context, op_type, a, axis, output_dtype)
  axis = axis or 0
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("eager " .. op_type:lower() .. " expects an integer axis", 3)
  end
  local dtype = Tensor.dtype_info(output_dtype or "int64")
  local dimension = context:constant(axis, "int32")
  return context:execute(op_type, { a, dimension },
    { attrs = { T = a:dtype(), Tidx = "int32", output_type = dtype } })[1]
end

function Context:argmax(a, axis, output_dtype)
  return arg_reduction(self, "ArgMax", a, axis, output_dtype)
end

function Context:argmin(a, axis, output_dtype)
  return arg_reduction(self, "ArgMin", a, axis, output_dtype)
end

function Context:relu(a)
  return unary(self, "Relu", a)
end

function Context:relu6(a)
  return unary(self, "Relu6", a)
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

function Context:floor(a)
  return unary(self, "Floor", a)
end

function Context:ceil(a)
  return unary(self, "Ceil", a)
end

function Context:round(a)
  return unary(self, "Round", a)
end

function Context:sign(a)
  return unary(self, "Sign", a)
end

function Context:rsqrt(a)
  return unary(self, "Rsqrt", a)
end

function Context:reciprocal(a)
  return unary(self, "Reciprocal", a)
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

function Context:log_softmax(a)
  return unary(self, "LogSoftmax", a)
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

local function integer_axis(axis, label)
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error("eager " .. label .. " expects an integer axis", 3)
  end
  return axis
end

local function integer_table(value, label)
  if type(value) ~= "table" or #value == 0 then
    error("eager " .. label .. " expects a non-empty integer table", 3)
  end
  for _, entry in ipairs(value) do
    if type(entry) ~= "number" or entry % 1 ~= 0 then
      error("eager " .. label .. " values must be integers", 3)
    end
  end
  return value
end

function Context:concat(inputs, axis)
  if type(inputs) ~= "table" or #inputs < 2 then
    error("eager concat expects at least two inputs", 2)
  end
  axis = integer_axis(axis or 0, "concat")
  local axis_tensor = self:constant(axis, "int32")
  local arguments = {}
  for i, input in ipairs(inputs) do arguments[i] = input end
  arguments[#arguments + 1] = axis_tensor
  return self:execute("ConcatV2", arguments,
    { attrs = { T = inputs[1]:dtype(), Tidx = "int32",
      N = { int = #inputs } } })[1]
end

function Context:gather(params, indices, axis)
  axis = integer_axis(axis or 0, "gather")
  local axis_tensor = self:constant(axis, "int32")
  return self:execute("GatherV2", { params, indices, axis_tensor },
    { attrs = { Tparams = params:dtype(), Tindices = indices:dtype(),
      Taxis = "int32", batch_dims = { int = 0 } } })[1]
end

function Context:slice(a, begin, size)
  begin = integer_table(begin, "slice begin")
  if type(size) ~= "table" or #size ~= #begin then
    error("eager slice expects matching begin and size tables", 2)
  end
  for _, value in ipairs(size) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < -1 then
      error("eager slice size values must be integers >= -1", 2)
    end
  end
  local begin_tensor = self:constant(begin, "int32")
  local size_tensor = self:constant(size, "int32")
  return self:execute("Slice", { a, begin_tensor, size_tensor },
    { attrs = { T = a:dtype(), Index = "int32" } })[1]
end

function Context:pad(a, paddings, mode)
  if type(paddings) ~= "table" or #paddings == 0 then
    error("eager pad expects a non-empty paddings table", 2)
  end
  for _, row in ipairs(paddings) do
    if type(row) ~= "table" or #row ~= 2 then
      error("eager pad paddings must be pairs of integers", 2)
    end
    for _, amount in ipairs(row) do
      if type(amount) ~= "number" or amount % 1 ~= 0 or amount < 0 then
        error("eager pad amounts must be non-negative integers", 2)
      end
    end
  end
  mode = string.upper(mode or "CONSTANT")
  if mode ~= "CONSTANT" and mode ~= "REFLECT" and mode ~= "SYMMETRIC" then
    error("eager pad mode must be CONSTANT, REFLECT, or SYMMETRIC", 2)
  end
  local padding_tensor = self:constant(paddings, "int32")
  local attrs = { T = a:dtype(), Tpaddings = "int32" }
  local op_type = "Pad"
  if mode ~= "CONSTANT" then
    op_type = "MirrorPad"
    attrs.mode = { string = mode }
  end
  return self:execute(op_type, { a, padding_tensor }, { attrs = attrs })[1]
end

function Context:tile(a, multiples)
  multiples = integer_table(multiples, "tile multiples")
  for _, value in ipairs(multiples) do
    if value < 0 then
      error("eager tile multiples must be non-negative integers", 2)
    end
  end
  local multiples_tensor = self:constant(multiples, "int32")
  return self:execute("Tile", { a, multiples_tensor },
    { attrs = { T = a:dtype(), Tmultiples = "int32" } })[1]
end

function Context:where(condition, x, y)
  return self:execute("SelectV2", { condition, x, y },
    { attrs = { T = x:dtype() } })[1]
end

function Context:reverse(a, axis)
  if type(axis) == "number" then axis = { axis } end
  axis = integer_table(axis, "reverse axis")
  local axis_tensor = self:constant(axis, "int32")
  return self:execute("ReverseV2", { a, axis_tensor },
    { attrs = { T = a:dtype(), Tidx = "int32" } })[1]
end

function Context:split(a, num_or_size_splits, axis)
  axis = integer_axis(axis or 0, "split")
  local axis_tensor = self:constant(axis, "int32")
  if type(num_or_size_splits) == "number" then
    if num_or_size_splits % 1 ~= 0 or num_or_size_splits < 1 then
      error("eager split expects a positive integer split count", 2)
    end
    return self:execute("Split", { axis_tensor, a },
      { attrs = { T = a:dtype(), num_split = { int = num_or_size_splits } },
        num_outputs = num_or_size_splits })
  elseif type(num_or_size_splits) == "table" then
    if #num_or_size_splits == 0 then
      error("eager split expects a non-empty size table", 2)
    end
    for _, size in ipairs(num_or_size_splits) do
      if type(size) ~= "number" or size % 1 ~= 0 or size < 0 then
        error("eager split sizes must be non-negative integers", 2)
      end
    end
    local sizes = self:constant(num_or_size_splits, "int64")
    return self:execute("SplitV", { a, sizes, axis_tensor },
      { attrs = { T = a:dtype(), Tlen = "int64", Tidx = "int32",
        num_split = { int = #num_or_size_splits } },
        num_outputs = #num_or_size_splits })
  end
  error("eager split expects a split count or size table", 2)
end

function Context:stack(inputs, axis)
  if type(inputs) ~= "table" or #inputs == 0 then
    error("eager stack expects a non-empty input list", 2)
  end
  axis = integer_axis(axis or 0, "stack")
  return self:execute("Pack", inputs,
    { attrs = { T = inputs[1]:dtype(), N = { int = #inputs },
      axis = { int = axis } } })[1]
end

function Context:unstack(a, num, axis)
  if type(num) ~= "number" or num % 1 ~= 0 or num < 1 then
    error("eager unstack expects a positive integer count", 2)
  end
  axis = integer_axis(axis or 0, "unstack")
  return self:execute("Unpack", { a },
    { attrs = { T = a:dtype(), num = { int = num }, axis = { int = axis } },
      num_outputs = num })
end

function Context:one_hot(indices, depth, options)
  options = options or {}
  if type(depth) ~= "number" or depth % 1 ~= 0 or depth < 1 then
    error("eager one_hot expects a positive integer depth", 2)
  end
  local dtype = Tensor.dtype_info(options.dtype or "float32")
  local depth_tensor = self:constant(depth, "int32")
  local on = self:constant(options.on_value or 1, dtype)
  local off = self:constant(options.off_value or 0, dtype)
  return self:execute("OneHot", { indices, depth_tensor, on, off },
    { attrs = { T = dtype, TI = indices:dtype(),
      axis = { int = options.axis or -1 } } })[1]
end

function Context:clip_by_value(a, minimum, maximum)
  if type(minimum) ~= "number" or type(maximum) ~= "number" then
    error("eager clip_by_value expects numeric minimum and maximum", 2)
  end
  local dtype = a:dtype()
  return self:execute("ClipByValue", { a, self:constant(minimum, dtype),
    self:constant(maximum, dtype) }, { attrs = { T = dtype } })[1]
end

function Context:close()
  if self._closed then return end
  self._closed = true
  if self._refs <= 0 then self:_destroy() end
end

Eager.Context = Context
return Eager
