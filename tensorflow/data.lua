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

function Dataset:size()
  return #self._x
end

function Dataset:batch(size)
  if type(size) ~= "number" or size < 1 then
    error("batch size must be positive", 2)
  end
  self._batch_size = size
  return self
end

function Dataset:shuffle(seed)
  self._seed = seed
  self._shuffle = true
  return self
end

function Dataset:each()
  local size = self._batch_size or #self._x
  local order = {}
  for i = 1, #self._x do order[i] = i end
  if self._shuffle then
    if self._seed ~= nil then math.randomseed(self._seed) end
    for i = #order, 2, -1 do
      local j = math.random(i)
      order[i], order[j] = order[j], order[i]
    end
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
    return { x = batch_x, y = self._y ~= nil and batch_y or nil }
  end
end

Dataset.Dataset = Dataset

return Dataset
