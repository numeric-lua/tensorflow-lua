--
-- Pure Lua learning-rate schedules.  Each factory returns a function of the
-- zero-based step/epoch and never touches a graph.
--
local Schedules = {}

local function positive(value, name)
  if type(value) ~= "number" or value <= 0 then
    error("schedules." .. name .. " must be positive", 3)
  end
  return value
end

function Schedules.exponential_decay(initial_rate, decay_steps, decay_rate,
    staircase)
  positive(initial_rate, "exponential_decay initial_rate")
  positive(decay_steps, "exponential_decay decay_steps")
  positive(decay_rate, "exponential_decay decay_rate")
  return function(step)
    local exponent = (step or 0) / decay_steps
    if staircase then exponent = math.floor(exponent) end
    return initial_rate * (decay_rate ^ exponent)
  end
end

function Schedules.cosine_decay(initial_rate, decay_steps, alpha)
  positive(initial_rate, "cosine_decay initial_rate")
  positive(decay_steps, "cosine_decay decay_steps")
  alpha = alpha or 0
  if type(alpha) ~= "number" or alpha < 0 or alpha > 1 then
    error("schedules.cosine_decay alpha must be in [0, 1]", 3)
  end
  return function(step)
    local progress = math.min(1, (step or 0) / decay_steps)
    local cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return (initial_rate - alpha * initial_rate) * cosine + alpha * initial_rate
  end
end

function Schedules.polynomial_decay(initial_rate, decay_steps, final_rate,
    power)
  positive(initial_rate, "polynomial_decay initial_rate")
  positive(decay_steps, "polynomial_decay decay_steps")
  final_rate = final_rate or 0
  if type(final_rate) ~= "number" or final_rate < 0 then
    error("schedules.polynomial_decay final_rate must be non-negative", 3)
  end
  power = power or 1
  positive(power, "polynomial_decay power")
  return function(step)
    local progress = math.min(1, (step or 0) / decay_steps)
    return (initial_rate - final_rate) * ((1 - progress) ^ power) + final_rate
  end
end

function Schedules.piecewise_constant(initial_rate, boundaries, values)
  positive(initial_rate, "piecewise_constant initial_rate")
  if type(boundaries) ~= "table" or type(values) ~= "table" then
    error("schedules.piecewise_constant expects boundary and value tables", 3)
  end
  if #values ~= #boundaries + 1 then
    error("schedules.piecewise_constant needs one more value than boundary", 3)
  end
  for index = 1, #boundaries do
    if type(boundaries[index]) ~= "number" or boundaries[index] < 0 or
        (index > 1 and boundaries[index] <= boundaries[index - 1]) then
      error("schedules.piecewise_constant boundaries must increase", 3)
    end
  end
  for index = 1, #values do
    if type(values[index]) ~= "number" or values[index] < 0 then
      error("schedules.piecewise_constant values must be non-negative", 3)
    end
  end
  return function(step)
    step = step or 0
    for index = 1, #boundaries do
      if step < boundaries[index] then return values[index] end
    end
    return values[#values]
  end
end

function Schedules.constant(rate)
  positive(rate, "constant")
  return function() return rate end
end

return Schedules
