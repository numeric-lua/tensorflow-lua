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

Dataset.Dataset = Dataset

return Dataset
