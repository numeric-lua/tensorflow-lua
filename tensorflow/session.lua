--
-- TF_Session wrapper, feeds/fetches, and SavedModel loading.
--
local ffi = require("ffi")
local capi = require("tensorflow.capi")
local Tensor = require("tensorflow.tensor")
local Proto = require("tensorflow.protobuf")

local Session = {}
Session.__index = Session

local function parse_output(graph, spec)
  if type(spec) == "table" and spec.oper ~= nil then return spec end
  local name, index = spec, 0
  local base, suffix = spec:match("^(.-):(%d+)$")
  if base then name, index = base, tonumber(suffix) end
  return { oper = graph:operation(name), index = index,
    name = name .. ":" .. tostring(index) }
end

local function output_array(graph, specs)
  local outputs = ffi.new("TF_Output[?]", #specs)
  for i, spec in ipairs(specs) do
    local resolved = parse_output(graph, spec)
    if type(resolved) == "table" then
      outputs[i - 1].oper = resolved.oper
      outputs[i - 1].index = resolved.index or 0
    else
      outputs[i - 1].oper = resolved[1]
      outputs[i - 1].index = resolved[2]
    end
  end
  return outputs
end

local function install_finalizer(pointer, graph_handle)
  local function finalize(value)
    local keepalive = graph_handle
    if keepalive == nil then return end
    local lib = capi.load()
    local status = lib.TF_NewStatus()
    lib.TF_CloseSession(value, status)
    lib.TF_DeleteStatus(status)
    status = lib.TF_NewStatus()
    lib.TF_DeleteSession(value, status)
    lib.TF_DeleteStatus(status)
    capi.released.sessions = capi.released.sessions + 1
  end
  return ffi.gc(pointer, finalize)
end

function Session.new(graph, options)
  local lib = capi.load()
  local opts = capi.require(lib.TF_NewSessionOptions(),
    "TF_NewSessionOptions")
  if options and options.target then lib.TF_SetTarget(opts, options.target) end
  local status = capi.status()
  local graph_handle = graph:handle()
  local ptr = lib.TF_NewSession(graph_handle, opts, status)
  local ok, err = pcall(capi.check, status, "session creation")
  lib.TF_DeleteSessionOptions(opts)
  if not ok or ptr == nil then error(err or "TF_NewSession failed", 2) end
  local handle = install_finalizer(ptr, graph_handle)
  return setmetatable({ _handle = handle, _graph = graph,
    _graph_handle = graph_handle }, Session)
end

function Session.load_saved_model(path, tags, options)
  local lib = capi.load()
  local Graph = require("tensorflow.graph")
  local graph = Graph.new()
  local opts = capi.require(lib.TF_NewSessionOptions(),
    "TF_NewSessionOptions")
  if options and options.target then lib.TF_SetTarget(opts, options.target) end
  tags = tags or { "serve" }
  if type(tags) ~= "table" or #tags == 0 then
    error("SavedModel tags must be a non-empty table of strings", 3)
  end
  for _, tag in ipairs(tags) do
    if type(tag) ~= "string" then
      error("SavedModel tags must be strings", 3)
    end
  end
  local tag_array = ffi.new("const char*[?]", #tags)
  for i = 1, #tags do tag_array[i - 1] = ffi.cast("const char*", tags[i]) end
  local meta = capi.require(lib.TF_NewBuffer(), "TF_NewBuffer")
  local status = capi.status()
  local ptr = lib.TF_LoadSessionFromSavedModel(opts, nil, path, tag_array,
    #tags, graph:handle(), meta, status)
  local ok, err = pcall(capi.check, status, "SavedModel load")
  lib.TF_DeleteSessionOptions(opts)
  if not ok or ptr == nil then
    lib.TF_DeleteBuffer(meta)
    error(err or "TF_LoadSessionFromSavedModel failed", 2)
  end
  local struct = lib.TF_GetBuffer(meta)
  local bytes = struct.length > 0 and struct.data ~= nil and
    ffi.string(struct.data, struct.length) or ""
  local decode_ok, signatures = pcall(Proto.saved_model_signatures, bytes)
  lib.TF_DeleteBuffer(meta)
  local graph_handle = graph:handle()
  local handle = install_finalizer(ptr, graph_handle)
  if not decode_ok then
    ffi.gc(handle, nil)
    local close_status = capi.status()
    lib.TF_CloseSession(ptr, close_status)
    capi.check(close_status, "SavedModel cleanup close")
    local delete_status = capi.status()
    lib.TF_DeleteSession(ptr, delete_status)
    capi.check(delete_status, "SavedModel cleanup delete")
    graph:delete()
    error(signatures, 2)
  end
  local session = setmetatable({ _handle = handle, _graph = graph,
    _graph_handle = graph_handle, signatures = signatures }, Session)
  return session, graph, signatures
end

function Session:run(feeds, fetches, targets)
  local lib = capi.load()
  if self._handle == nil then error("session is closed", 2) end
  if type(fetches) ~= "table" and type(fetches) ~= "string" then
    error("run expects a fetch name or a list of fetch names", 2)
  end
  local named_fetches = false
  if type(fetches) == "table" then
    local has_targets = type(targets) == "string" or
      (type(targets) == "table" and #targets > 0)
    if next(fetches) == nil and not has_targets then
      error("run expects a non-empty fetch list or named map", 2)
    end
    named_fetches = fetches[1] == nil
  end
  local function label(spec)
    if type(spec) == "table" then return spec.name or "output" end
    return spec
  end
  local fetch_specs, fetch_names = {}, {}
  if type(fetches) == "string" then
    fetch_specs[1] = fetches
    fetch_names[1] = fetches
  elseif named_fetches then
    for name, spec in pairs(fetches) do
      fetch_names[#fetch_names + 1] = name
      fetch_specs[#fetch_specs + 1] = spec
    end
  else
    for i, spec in ipairs(fetches) do
      fetch_names[i] = label(spec)
      fetch_specs[i] = spec
    end
  end

  local feed_specs, feed_values = {}, {}
  for name, value in pairs(feeds or {}) do
    if type(name) ~= "string" and not (type(name) == "table" and name.oper) then
      error("feed names must be strings or graph outputs", 2)
    end
    if getmetatable(value) ~= Tensor then
      local dtype = type(name) == "table" and name.dtype or nil
      value = Tensor.new(value, { dtype = dtype })
    end
    feed_specs[#feed_specs + 1] = name
    feed_values[#feed_values + 1] = value
  end

  local input_outputs = output_array(self._graph, feed_specs)
  local input_tensors = ffi.new("TF_Tensor*[?]", #feed_values)
  for i, value in ipairs(feed_values) do
    input_tensors[i - 1] = ffi.cast("TF_Tensor*", value:handle())
  end

  local output_outputs = output_array(self._graph, fetch_specs)
  local output_tensors = ffi.new("TF_Tensor*[?]", #fetch_specs)
  for i = 0, #fetch_specs - 1 do output_tensors[i] = nil end

  local target_specs = targets or {}
  if type(target_specs) == "string" then target_specs = { target_specs } end
  local target_ops = nil
  if #target_specs > 0 then
    target_ops = ffi.new("TF_Operation*[?]", #target_specs)
    for i, target in ipairs(target_specs) do
      if type(target) == "table" and target.oper then
        target_ops[i - 1] = target.oper
      else
        target_ops[i - 1] = self._graph:operation(target)
      end
    end
  end

  local target_pointer = target_ops and ffi.cast(
    "const TF_Operation* const*", target_ops) or nil
  local status = capi.status()
  lib.TF_SessionRun(self:handle(), nil,
    input_outputs, input_tensors, #feed_values,
    output_outputs, output_tensors, #fetch_specs,
    target_pointer, #target_specs, nil, status)
  local ok, err = pcall(capi.check, status, "session run")
  if not ok then
    for i = 0, #fetch_specs - 1 do
      if output_tensors[i] ~= nil then lib.TF_DeleteTensor(output_tensors[i]) end
    end
    error(err, 2)
  end

  local wrapped = {}
  for i = 1, #fetch_names do
    if output_tensors[i - 1] == nil then
      for j = i, #fetch_names do
        if output_tensors[j - 1] ~= nil then
          lib.TF_DeleteTensor(output_tensors[j - 1])
        end
      end
      error('session returned a NULL "' .. fetch_names[i] .. '" tensor', 2)
    end
    local wrap_ok, value = pcall(Tensor.wrap, output_tensors[i - 1])
    if not wrap_ok then
      for j = i, #fetch_names do
        if output_tensors[j - 1] ~= nil then
          lib.TF_DeleteTensor(output_tensors[j - 1])
        end
      end
      error(value, 2)
    end
    wrapped[i] = value
  end
  if type(fetches) == "string" then return wrapped[1] end
  if named_fetches then
    local results = {}
    for i, name in ipairs(fetch_names) do results[name] = wrapped[i] end
    return results
  end
  return wrapped
end

function Session:run_signature(name, feeds)
  local signature = self.signatures and self.signatures[name]
  if signature == nil then
    error("SavedModel has no signature '" .. tostring(name) .. "'", 2)
  end
  local mapped_feeds = {}
  for key, value in pairs(feeds or {}) do
    local info = signature.inputs[key]
    if info == nil then
      error("signature '" .. name .. "' has no input '" .. tostring(key) .. "'", 2)
    end
    mapped_feeds[info.name] = value
  end
  local keys, specs = {}, {}
  for key, info in pairs(signature.outputs) do
    keys[#keys + 1] = key
    specs[#specs + 1] = info.name
  end
  local values = self:run(mapped_feeds, specs)
  local result = {}
  for i, key in ipairs(keys) do result[key] = values[i] end
  return result
end

function Session:handle()
  if self._handle == nil then error("session is closed", 2) end
  return self._handle
end

function Session:close()
  if self._handle == nil then return end
  local handle = self._handle
  self._handle = nil
  ffi.gc(handle, nil)
  local lib = capi.load()
  local close_status = capi.status()
  lib.TF_CloseSession(handle, close_status)
  local close_ok, close_err = pcall(capi.check, close_status, "session close")
  local delete_status = capi.status()
  lib.TF_DeleteSession(handle, delete_status)
  capi.released.sessions = capi.released.sessions + 1
  local delete_ok, delete_err = pcall(capi.check, delete_status,
    "session delete")
  if not close_ok then error(close_err, 2) end
  if not delete_ok then error(delete_err, 2) end
end

return Session
