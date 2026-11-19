--
-- Minimal dependency-free dataset batching.
--
local Dataset = {}
Dataset.__index = Dataset

function Dataset.new(x, y)
  if type(x) ~= "table" then error("Dataset.new expects a table", 2) end
  if y ~= nil and (#x ~= #y) then
    error("Dataset x and y must have the same length", 2)
  end
  return setmetatable({ _x = x, _y = y, _batch_size = nil, _shuffle = false },
    Dataset)
end

function Dataset:data()
  return self._x, self._y, self._batch_size, self._shuffle, self._seed
end

function Dataset:map(fn)
  if type(fn) ~= "function" then error("map expects a function", 2) end
  local mapped_x, mapped_y = {}, {}
  for i = 1, #self._x do
    local x, y = fn(self._x[i], self._y and self._y[i] or nil)
    mapped_x[i] = x
    if self._y ~= nil then mapped_y[i] = y end
  end
  return Dataset.new(mapped_x, self._y ~= nil and mapped_y or nil)
end

function Dataset:filter(fn)
  if type(fn) ~= "function" then error("filter expects a function", 2) end
  local filtered_x, filtered_y = {}, {}
  for i = 1, #self._x do
    local y = self._y and self._y[i] or nil
    if fn(self._x[i], y) then
      filtered_x[#filtered_x + 1] = self._x[i]
      if self._y ~= nil then filtered_y[#filtered_y + 1] = y end
    end
  end
  return Dataset.new(filtered_x, self._y ~= nil and filtered_y or nil)
end

function Dataset:split(fraction)
  if type(fraction) ~= "number" or fraction <= 0 or fraction >= 1 then
    error("split fraction must be between 0 and 1", 2)
  end
  local cut = math.floor(#self._x * fraction)
  if cut < 1 or cut >= #self._x then
    error("split fraction produces an empty partition", 2)
  end
  local train_x, train_y, validation_x, validation_y = {}, {}, {}, {}
  for i = 1, cut do
    train_x[i] = self._x[i]
    if self._y ~= nil then train_y[i] = self._y[i] end
  end
  for i = cut + 1, #self._x do
    validation_x[#validation_x + 1] = self._x[i]
    if self._y ~= nil then
      validation_y[#validation_y + 1] = self._y[i]
    end
  end
  return Dataset.new(train_x, self._y ~= nil and train_y or nil),
    Dataset.new(validation_x, self._y ~= nil and validation_y or nil)
end

function Dataset:size()
  return #self._x
end

function Dataset:repeat_times(count)
  if count ~= nil and (type(count) ~= "number" or count < 1) then
    error("repeat count must be positive or nil", 2)
  end
  self._repeats = count
  return self
end

function Dataset:take(count)
  if type(count) ~= "number" or count < 0 then
    error("take count must be non-negative", 2)
  end
  self._take = count
  return self
end

function Dataset:batch(size, options)
  if type(size) ~= "number" or size < 1 then
    error("batch size must be positive", 2)
  end
  options = options or {}
  self._batch_size = size
  self._drop_last = options.drop_last or false
  return self
end

function Dataset:shuffle(seed)
  self._seed = seed
  self._shuffle = true
  return self
end

function Dataset:each()
  local size = self._batch_size or #self._x
  local repeats = self._repeats or 1
  local order = {}
  for _ = 1, repeats do
    if self._shuffle then
      if self._seed ~= nil then math.randomseed(self._seed) end
      local pass = {}
      for i = 1, #self._x do pass[i] = i end
      for i = #pass, 2, -1 do
        local j = math.random(i)
        pass[i], pass[j] = pass[j], pass[i]
      end
      for _, index in ipairs(pass) do order[#order + 1] = index end
    else
      for i = 1, #self._x do order[#order + 1] = i end
    end
  end
  if self._take ~= nil and #order > self._take then
    for i = #order, self._take + 1, -1 do order[i] = nil end
  end
  local position = 1
  return function()
    if position > #order then return nil end
    local last = math.min(position + size - 1, #order)
    local batch_x, batch_y = {}, {}
    for i = position, last do
      local index = order[i]
      batch_x[#batch_x + 1] = self._x[index]
      if self._y ~= nil then batch_y[#batch_y + 1] = self._y[index] end
    end
    position = last + 1
    if self._drop_last and #batch_x < size then return nil end
    return { x = batch_x, y = self._y ~= nil and batch_y or nil }
  end
end

-- Tensor-backed dataset.  Batches are produced without converting the full
-- tensors to Lua tables.
local TensorDataset = {}
TensorDataset.__index = TensorDataset

function Dataset.from_tensor(x, y)
  local Tensor = require("tensorflow.tensor")
  if getmetatable(x) ~= Tensor then
    error("Dataset.from_tensor expects a Tensor input", 2)
  end
  if y ~= nil and getmetatable(y) ~= Tensor then
    error("Dataset.from_tensor target must be a Tensor", 2)
  end
  if y ~= nil and x:shape()[1] ~= y:shape()[1] then
    error("Dataset tensor batch sizes differ", 2)
  end
  return setmetatable({
    _x = x, _y = y, _tensor_dataset = true, _batch_size = nil,
    _shuffle = false,
  }, TensorDataset)
end

function TensorDataset:tensor_data()
  return self._x, self._y, self._batch_size, self._shuffle, self._seed
end

function TensorDataset:size()
  return self._x:shape()[1]
end

function TensorDataset:batch(size, options)
  if type(size) ~= "number" or size < 1 then
    error("batch size must be positive", 2)
  end
  options = options or {}
  self._batch_size = size
  self._drop_last = options.drop_last or false
  return self
end

function TensorDataset:shuffle(seed)
  self._shuffle = true
  self._seed = seed
  return self
end

function TensorDataset:take(count)
  self._take = count
  return self
end

function TensorDataset:each()
  local count = self:size()
  local size = self._batch_size or count
  local order = {}
  for index = 1, count do order[index] = index end
  if self._shuffle then
    if self._seed ~= nil then math.randomseed(self._seed) end
    for index = count, 2, -1 do
      local swap = math.random(index)
      order[index], order[swap] = order[swap], order[index]
    end
  end
  if self._take ~= nil and #order > self._take then
    for index = #order, self._take + 1, -1 do order[index] = nil end
  end
  local position = 1
  return function()
    if position > #order then return nil end
    local last = math.min(position + size - 1, #order)
    local indices = {}
    for index = position, last do
      indices[#indices + 1] = order[index]
    end
    local batch_x = self._x:take_rows(indices)
    local batch_y = self._y and self._y:take_rows(indices) or nil
    position = last + 1
    if self._drop_last and #indices < size then
      batch_x:destroy(); if batch_y then batch_y:destroy() end
      return nil
    end
    return { x = batch_x, y = batch_y }
  end
end

Dataset.TensorDataset = TensorDataset

Dataset.Dataset = Dataset

return Dataset
