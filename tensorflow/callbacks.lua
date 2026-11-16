--
-- Training callbacks for tf.Sequential.
--
local Callbacks = {}

local function monitored(history, monitor)
  local values = history[monitor]
  if values == nil or #values == 0 then
    error("callback monitor '" .. monitor .. "' is missing from history", 3)
  end
  return values[#values]
end

local function improved(mode, value, best, min_delta)
  if best == nil then return true end
  if mode == "max" then return value > best + min_delta end
  return value < best - min_delta
end

function Callbacks.EarlyStopping(options)
  options = options or {}
  local monitor = options.monitor or "val_loss"
  local callback = {
    monitor = monitor,
    mode = options.mode or (monitor:find("loss") and "min" or "max"),
    patience = options.patience or 0,
    min_delta = options.min_delta or 0,
    restore_best_weights = options.restore_best_weights or false,
    best = nil,
    wait = 0,
    best_weights = nil,
    stopped_epoch = nil,
  }
  function callback.on_train_begin(_, model)
    callback.best = nil
    callback.wait = 0
    callback.stopped_epoch = nil
    callback.best_weights = nil
    if callback.restore_best_weights then
      if model == nil or type(model.get_weights) ~= "function" or
          type(model.set_weights) ~= "function" then
        error("restore_best_weights requires get_weights/set_weights", 3)
      end
    end
  end
  function callback.on_epoch_end(history, epoch, model)
    local value = monitored(history, callback.monitor)
    if improved(callback.mode, value, callback.best, callback.min_delta) then
      callback.best = value
      callback.wait = 0
      if callback.restore_best_weights then
        callback.best_weights = model:get_weights()
      end
    else
      callback.wait = callback.wait + 1
      if callback.wait > callback.patience then
        history.stop_training = true
        callback.stopped_epoch = epoch
        if callback.restore_best_weights and callback.best_weights ~= nil then
          model:set_weights(callback.best_weights)
        end
      end
    end
  end
  function callback.on_train_end()
    callback.best_weights = nil
  end
  return callback
end

function Callbacks.ReduceLROnPlateau(options)
  options = options or {}
  local monitor = options.monitor or "val_loss"
  local callback = {
    monitor = monitor,
    mode = options.mode or (monitor:find("loss") and "min" or "max"),
    factor = options.factor or 0.1,
    patience = options.patience or 10,
    min_delta = options.min_delta or 0,
    min_lr = options.min_lr or 0,
    cooldown = options.cooldown or 0,
    best = nil,
    wait = 0,
    cooldown_left = 0,
    learning_rate = nil,
  }
  if callback.factor <= 0 or callback.factor >= 1 then
    error("ReduceLROnPlateau factor must be in (0, 1)", 3)
  end
  function callback.on_train_begin()
    callback.best = nil
    callback.wait = 0
    callback.cooldown_left = 0
    callback.learning_rate = nil
  end
  function callback.on_epoch_end(history, _, model)
    if model == nil or model._current_lr == nil then
      error("ReduceLROnPlateau requires a model with _current_lr", 3)
    end
    local value = monitored(history, callback.monitor)
    if improved(callback.mode, value, callback.best, callback.min_delta) then
      callback.best = value
      callback.wait = 0
      if callback.cooldown_left > 0 then
        callback.cooldown_left = callback.cooldown_left - 1
      end
      return
    end
    callback.wait = callback.wait + 1
    if callback.cooldown_left > 0 then
      callback.cooldown_left = callback.cooldown_left - 1
      return
    end
    if callback.wait > callback.patience then
      local next_lr = math.max(model._current_lr * callback.factor,
        callback.min_lr)
      model._current_lr = next_lr
      callback.learning_rate = next_lr
      callback.wait = 0
      callback.cooldown_left = callback.cooldown
    end
  end
  return callback
end

function Callbacks.ModelCheckpoint(path, options)
  options = options or {}
  if type(path) ~= "string" or path == "" then
    error("ModelCheckpoint requires a path", 2)
  end
  local monitor = options.monitor or "val_loss"
  local callback = {
    path = path,
    monitor = monitor,
    mode = options.mode or (monitor:find("loss") and "min" or "max"),
    save_best_only = options.save_best_only or false,
    save_training = options.save_training or false,
    min_delta = options.min_delta or 0,
    best = nil,
  }
  function callback.on_epoch_end(history, epoch, model)
    local value = monitored(history, callback.monitor)
    if callback.save_best_only and
        not improved(callback.mode, value, callback.best,
          callback.min_delta) then
      return
    end
    callback.best = value
    local resolved = callback.path:gsub("{epoch}", tostring(epoch))
    if callback.save_training then
      if type(model.save_training) ~= "function" then
        error("save_training checkpoint requires model:save_training", 3)
      end
      model:save_training(resolved)
    else
      model:save(resolved)
    end
  end
  return callback
end

return Callbacks
