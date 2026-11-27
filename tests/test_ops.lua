local tf = require("tensorflow")

test("runs typed unary, reduction, reshape, and argmax ops", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 3 }, "ox")
  local relu = tf.nn.relu(graph, x, "orelu")
  local mean = tf.reduce_mean(graph, relu, { 0 }, "omean")
  local shaped = tf.reshape(graph, relu, { 3, 1 }, "oshaped")
  local softmax = tf.nn.softmax(graph, x, "osoftmax")
  local argmax = tf.argmax(graph, x, 0, "oargmax")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { -2, 0, 2 } },
    { mean, shaped, softmax, argmax })
  assert_true(math.abs(result[1]:totable() - 2 / 3) < 1e-5)
  assert_eq(table.concat(result[2]:shape(), ","), "3,1")
  assert_eq(result[4]:totable(), 2)
  local probabilities = result[3]:totable()
  assert_true(probabilities[1] < probabilities[3])
  session:close()
end)

test("runs ArgMin and Relu6 graph ops", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 4 }, "ext_x")
  local clipped = tf.nn.relu6(graph, x, "ext_relu6")
  local minimum = tf.argmin(graph, x, 0, "ext_argmin", "float32", "int32")
  local maximum = tf.argmax(graph, x, 0, "ext_argmax", "float32", "int32")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { -3, 2, 9, 4 } },
    { clipped, minimum, maximum })
  assert_eq(result[1]:dtype(), "float32")
  assert_eq(table.concat(result[1]:totable(), ","), "0,2,6,4")
  assert_eq(result[2]:dtype(), "int32")
  assert_eq(result[2]:totable(), 0)
  assert_eq(result[3]:totable(), 2)
  assert_eq(graph:operation_info("ext_argmin").op_type, "ArgMin")
  assert_error(function()
    tf.argmin(graph, x, 0.5, "ext_bad_argmin")
  end, "integer axis")
  session:close()
end)

test("finds top-k values and in-top-k targets", function()
  local graph = tf.Graph.new()
  local logits = tf.placeholder(graph, "float32", { 4 }, "topk_x")
  local top = tf.top_k(graph, logits, 2, "topk")
  local values, indices = top[1], top[2]
  local predictions = tf.placeholder(graph, "float32", { 2, 4 }, "itk_p")
  local targets = tf.placeholder(graph, "int32", { 2 }, "itk_t")
  local hits = tf.in_top_k(graph, predictions, targets, 2, "itk")
  local session = tf.Session.new(graph)
  local result = session:run({
    [logits] = { 0.1, 0.5, 0.3, 0.2 },
    [predictions] = {
      { 0.1, 0.5, 0.3, 0.2 },
      { 0.9, 0.1, 0.0, 0.0 },
    },
    [targets] = { 1, 3 },
  }, { values, indices, hits })
  assert_eq(result[1]:totable()[1], 0.5)
  assert_true(math.abs(result[1]:totable()[2] - 0.3) < 1e-6)
  assert_eq(result[2]:dtype(), "int32")
  assert_eq(table.concat(result[2]:totable(), ","), "1,2")
  assert_eq(result[3]:dtype(), "bool")
  assert_eq(tostring(result[3]:totable()[1]), "true")
  assert_eq(tostring(result[3]:totable()[2]), "false")
  assert_error(function() tf.top_k(graph, logits, 0, "topk_bad") end,
    "positive")
  session:close()
end)

test("batches matrix multiplication and log-softmax", function()
  local graph = tf.Graph.new()
  local a = tf.placeholder(graph, "float32", { 2, 2, 3 }, "bmm_a")
  local b = tf.placeholder(graph, "float32", { 2, 3, 2 }, "bmm_b")
  local product = tf.batch_matmul(graph, a, b, "bmm")
  local transposed_a = tf.placeholder(graph, "float32", { 3, 2 }, "bmm_ta")
  local transposed = tf.placeholder(graph, "float32", { 3, 2 }, "bmm_tb")
  local adjoint = tf.batch_matmul(graph, transposed_a, transposed, "bmm_adj",
    nil, { adj_x = true })
  local logits = tf.placeholder(graph, "float32", { 3 }, "log_softmax_x")
  local log_softmax = tf.nn.log_softmax(graph, logits, "log_softmax")
  local logits64 = tf.placeholder(graph, "float64", { 3 }, "log_softmax_x64")
  local log_softmax64 = tf.nn.log_softmax(graph, logits64, "log_softmax64",
    "float64")
  local session = tf.Session.new(graph)
  local result = session:run({
    [a] = { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 1, 0, 0 }, { 0, 1, 0 } } },
    [b] = { { { 1, 0 }, { 0, 1 }, { 1, 1 } }, { { 1, 2 }, { 3, 4 }, { 5, 6 } } },
    [transposed_a] = { { 1, 2 }, { 3, 4 }, { 5, 6 } },
    [transposed] = { { 1, 0 }, { 0, 1 }, { 1, 1 } },
    [logits] = { 0, 1, 2 },
    [logits64] = { 0, 1, 2 },
  }, { product, adjoint, log_softmax, log_softmax64 })
  assert_eq(table.concat(result[1]:shape(), ","), "2,2,2")
  assert_eq(result[1]:totable()[1][1][1], 4)
  assert_eq(result[1]:totable()[1][2][2], 11)
  assert_eq(result[1]:totable()[2][1][2], 2)
  assert_eq(table.concat(result[2]:shape(), ","), "2,2")
  assert_eq(result[2]:totable()[1][1], 6)
  assert_eq(result[2]:totable()[2][2], 10)
  assert_eq(graph:attribute("bmm", "adj_x"), false)
  assert_eq(graph:attribute("bmm_adj", "adj_x"), true)
  local log_values = result[3]:totable()
  assert_true(math.abs(log_values[2] - log_values[1] - 1) < 1e-6)
  assert_true(math.abs(math.exp(log_values[1]) + math.exp(log_values[2]) +
    math.exp(log_values[3]) - 1) < 1e-6)
  assert_eq(result[4]:dtype(), "float64")
  assert_true(math.abs(result[4]:totable()[2] - log_values[2]) < 1e-6)
  assert_error(function()
    tf.batch_matmul(graph, a, b, "bmm_bad_adj", nil, { adj_x = "yes" })
  end, "boolean")
  session:close()
end)

test("fills tensors and builds numeric sequences", function()
  local graph = tf.Graph.new()
  local filled = tf.fill(graph, { 2, 3 }, 7, "fill_value")
  local scalar = tf.fill(graph, {}, 3, "fill_scalar")
  local sequence = tf.range(graph, 1, 6, 2, "range_value", "int32")
  local spaced = tf.linspace(graph, 0, 1, 5, "linspace_value")
  local session = tf.Session.new(graph)
  local result = session:run({}, { filled, scalar, sequence, spaced })
  assert_eq(table.concat(result[1]:shape(), ","), "2,3")
  assert_eq(result[1]:totable()[2][3], 7)
  assert_eq(#result[2]:shape(), 0)
  assert_eq(result[2]:totable(), 3)
  assert_eq(result[3]:dtype(), "int32")
  assert_eq(table.concat(result[3]:totable(), ","), "1,3,5")
  assert_eq(result[4]:dtype(), "float32")
  assert_eq(table.concat(result[4]:totable(), ","), "0,0.25,0.5,0.75,1")
  assert_error(function() tf.fill(graph, "dims", 1, "fill_bad_dims") end,
    "dims table")
  assert_error(function() tf.fill(graph, { -1 }, 1, "fill_negative") end,
    "non-negative")
  assert_error(function() tf.linspace(graph, 0, 1, 0, "linspace_zero") end,
    "positive")
  assert_error(function()
    tf.linspace(graph, 0, 1, 5, "linspace_int", "int32")
  end, "float32 or float64")
  session:close()
end)

test("builds zero and one tensors", function()
  local graph = tf.Graph.new()
  local zeros = tf.zeros(graph, { 2, 3 }, "zeros", "int32")
  local ones = tf.ones(graph, { 2, 2 }, "ones", "float64")
  local session = tf.Session.new(graph)
  local result = session:run({}, { zeros, ones })
  assert_eq(table.concat(result[1]:shape(), ","), "2,3")
  assert_eq(result[1]:dtype(), "int32")
  assert_eq(table.concat(result[1]:totable()[1], ","), "0,0,0")
  assert_eq(result[2]:dtype(), "float64")
  assert_eq(table.concat(result[2]:totable()[2], ","), "1,1")
  session:close()
end)

test("reduces variance and standard deviation", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 2, 3 }, "var_x")
  local variance = tf.reduce_variance(graph, matrix, { 1 }, "var")
  local stddev = tf.reduce_stddev(graph, matrix, { 1 }, "std", nil, true)
  local global = tf.reduce_variance(graph, matrix, { 0, 1 }, "var_global")
  local matrix64 = tf.placeholder(graph, "float64", { 2 }, "var_x64")
  local variance64 = tf.reduce_variance(graph, matrix64, { 0 }, "var64")
  local session = tf.Session.new(graph)
  local result = session:run({
    [matrix] = { { 1, 2, 3 }, { 2, 4, 6 } },
    [matrix64] = { 1, 3 },
  }, { variance, stddev, global, variance64 })
  assert_eq(table.concat(result[1]:shape(), ","), "2")
  assert_true(math.abs(result[1]:totable()[1] - 2 / 3) < 1e-6)
  assert_true(math.abs(result[1]:totable()[2] - 8 / 3) < 1e-6)
  assert_eq(table.concat(result[2]:shape(), ","), "2,1")
  assert_true(math.abs(result[2]:totable()[1][1] - math.sqrt(2 / 3)) < 1e-6)
  assert_true(math.abs(result[2]:totable()[2][1] - math.sqrt(8 / 3)) < 1e-6)
  assert_true(math.abs(result[3]:totable() - 8 / 3) < 1e-6)
  assert_eq(result[4]:dtype(), "float64")
  assert_eq(result[4]:totable(), 1)
  assert_error(function() tf.reduce_variance(graph, matrix, 1, "var_bad") end,
    "axes")
  assert_error(function()
    tf.reduce_stddev(graph, matrix, { 0.5 }, "std_bad")
  end, "integers")
  session:close()
end)

test("computes L1 and L2 tensor norms", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 2, 3 }, "norm_x")
  local l2 = tf.norm(graph, matrix, { 1 }, "l2", "norm_l2")
  local l1 = tf.norm(graph, matrix, { 1 }, "l1", "norm_l1")
  local global = tf.norm(graph, matrix, { 0, 1 }, nil, "norm_global")
  local session = tf.Session.new(graph)
  local result = session:run({ [matrix] = { { 1, 2, 3 }, { 4, 5, 6 } } },
    { l2, l1, global })
  assert_true(math.abs(result[1]:totable()[1] - math.sqrt(14)) < 1e-6)
  assert_true(math.abs(result[1]:totable()[2] - math.sqrt(77)) < 1e-6)
  assert_eq(table.concat(result[2]:totable(), ","), "6,15")
  assert_true(math.abs(result[3]:totable() - math.sqrt(91)) < 1e-6)
  session:close()
end)

test("adds tensor lists and computes stable log-sum-exp", function()
  local graph = tf.Graph.new()
  local a = tf.placeholder(graph, "float32", { 2 }, "lse_a")
  local b = tf.placeholder(graph, "float32", { 2 }, "lse_b")
  local summed = tf.add_n(graph, { a, b }, "lse_add")
  local matrix = tf.placeholder(graph, "float32", { 2, 2 }, "lse_matrix")
  local lse = tf.reduce_logsumexp(graph, matrix, { 1 }, "lse")
  local kept = tf.reduce_logsumexp(graph, matrix, { 1 }, "lse_kept", nil, true)
  local matrix64 = tf.placeholder(graph, "float64", { 2, 2 }, "lse_matrix64")
  local lse64 = tf.reduce_logsumexp(graph, matrix64, { 1 }, "lse64")
  local session = tf.Session.new(graph)
  local result = session:run({
    [a] = { 1, 2 },
    [b] = { 3, 4 },
    [matrix] = { { 0, 1 }, { 1000, 1000 } },
    [matrix64] = { { 0, 1 }, { 1000, 1000 } },
  }, { summed, lse, kept, lse64 })
  assert_eq(table.concat(result[1]:totable(), ","), "4,6")
  assert_true(math.abs(result[2]:totable()[1] - math.log(1 + math.exp(1))) <
    1e-5)
  assert_true(math.abs(result[2]:totable()[2] - (1000 + math.log(2))) < 1e-4)
  assert_eq(table.concat(result[3]:shape(), ","), "2,1")
  assert_eq(result[4]:dtype(), "float64")
  assert_true(math.abs(result[4]:totable()[2] - (1000 + math.log(2))) < 1e-10)
  assert_error(function() tf.add_n(graph, {}, "lse_empty") end,
    "non-empty")
  assert_error(function()
    tf.reduce_logsumexp(graph, matrix, { 0.5 }, "lse_bad_axis")
  end, "integers")
  session:close()
end)

test("typed ops reject invalid shapes", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", nil, "bx")
  assert_error(function() tf.reshape(graph, x, nil, "bad") end, "shape")
end)

test("builds one-hot tensors", function()
  local graph = tf.Graph.new()
  local indices = tf.placeholder(graph, "int32", { 2 }, "oh_indices")
  local encoded = tf.one_hot(graph, indices, 3, { name = "oh" })
  local session = tf.Session.new(graph)
  local values = session:run({ [indices] = { 0, 2 } },
    { encoded })[1]:totable()
  assert_eq(values[1][1], 1)
  assert_eq(values[1][2], 0)
  assert_eq(values[2][3], 1)
  session:close()
end)

test("casts and concatenates graph outputs", function()
  local graph = tf.Graph.new()
  local a = tf.placeholder(graph, "int32", { 2 }, "ca")
  local b = tf.placeholder(graph, "int32", { 2 }, "cb")
  local cast = tf.cast(graph, a, "int32", "float32", "ccast")
  local joined = tf.concat(graph, { a, b }, 0, "cconcat", "int32")
  local session = tf.Session.new(graph)
  local result = session:run({ [a] = { 1, 2 }, [b] = { 3, 4 } },
    { cast, joined })
  assert_eq(result[1]:dtype(), "float32")
  assert_eq(result[1]:totable()[2], 2)
  assert_eq(table.concat(result[2]:shape(), ","), "4")
  assert_eq(result[2]:totable()[4], 4)
  session:close()
end)

test("reduces sums, maxima, and minima", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 3 }, "red_x")
  local sum = tf.reduce_sum(graph, x, { 0 }, "red_sum")
  local maximum = tf.reduce_max(graph, x, { 0 }, "red_max")
  local minimum = tf.reduce_min(graph, x, { 0 }, "red_min")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { 1, 2, 3 } }, { sum, maximum, minimum })
  assert_eq(result[1]:totable(), 6)
  assert_eq(result[2]:totable(), 3)
  assert_eq(result[3]:totable(), 1)
  session:close()
end)

test("runs leaky ReLU and ELU", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2 }, "act_x")
  local leaky = tf.nn.leaky_relu(graph, x, "act_leaky", "float32", 0.1)
  local elu = tf.nn.elu(graph, x, "act_elu")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { -2, 3 } }, { leaky, elu })
  assert_true(math.abs(result[1]:totable()[1] + 0.2) < 1e-5)
  assert_true(math.abs(result[2]:totable()[1] - (math.exp(-2) - 1)) < 1e-5)
  session:close()
end)

test("runs Softsign and SELU activations", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 3 }, "ss_x")
  local softsign = tf.nn.softsign(graph, x, "ss_softsign")
  local selu = tf.nn.selu(graph, x, "ss_selu")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { -1, 0, 2 } }, { softsign, selu })
  local soft = result[1]:totable()
  assert_true(math.abs(soft[1] + 0.5) < 1e-6)
  assert_eq(soft[2], 0)
  assert_true(math.abs(soft[3] - 2 / 3) < 1e-6)
  local scale = 1.0507009873554805
  local alpha = 1.6732632423543772
  local activated = result[2]:totable()
  assert_true(math.abs(activated[1] - scale * alpha * (math.exp(-1) - 1)) <
    1e-5)
  assert_eq(activated[2], 0)
  assert_true(math.abs(activated[3] - scale * 2) < 1e-5)
  session:close()
end)

test("runs rounding, sign, reciprocal, and rsqrt ops", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 3 }, "math_x")
  local floor = tf.floor(graph, x, "math_floor")
  local ceil = tf.ceil(graph, x, "math_ceil")
  local round = tf.round(graph, x, "math_round")
  local sign = tf.sign(graph, x, "math_sign")
  local positive = tf.placeholder(graph, "float32", { 2 }, "math_positive")
  local reciprocal = tf.reciprocal(graph, positive, "math_reciprocal")
  local rsqrt = tf.rsqrt(graph, positive, "math_rsqrt")
  local float64 = tf.placeholder(graph, "float64", { 2 }, "math_float64")
  local rounded64 = tf.round(graph, float64, "math_round64")
  local session = tf.Session.new(graph)
  local result = session:run({
    [x] = { -1.4, 0.6, 2.7 },
    [positive] = { 4, 9 },
    [float64] = { 1.2, 2.8 },
  }, { floor, ceil, round, sign, reciprocal, rsqrt, rounded64 })
  assert_eq(table.concat(result[1]:totable(), ","), "-2,0,2")
  assert_eq(table.concat(result[2]:totable(), ","), "-1,1,3")
  assert_eq(table.concat(result[3]:totable(), ","), "-1,1,3")
  assert_eq(table.concat(result[4]:totable(), ","), "-1,1,1")
  assert_eq(result[5]:totable()[1], 0.25)
  assert_true(math.abs(result[5]:totable()[2] - 1 / 9) < 1e-6)
  assert_true(math.abs(result[6]:totable()[1] - 0.5) < 1e-6)
  assert_true(math.abs(result[6]:totable()[2] - 1 / 3) < 1e-6)
  assert_eq(result[7]:dtype(), "float64")
  assert_eq(table.concat(result[7]:totable(), ","), "1,3")
  session:close()
end)

test("runs error functions and predicates", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 5 }, "pred_x")
  local erf = tf.erf(graph, x, "pred_erf")
  local erfc = tf.erfc(graph, x, "pred_erfc")
  local is_nan = tf.is_nan(graph, x, "pred_nan")
  local is_inf = tf.is_inf(graph, x, "pred_inf")
  local is_finite = tf.is_finite(graph, x, "pred_finite")
  local nan, inf = 0 / 0, math.huge
  assert_true(nan ~= nan)
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { 0, 1, nan, inf, -inf } },
    { erf, erfc, is_nan, is_inf, is_finite })
  assert_eq(result[1]:totable()[1], 0)
  assert_true(math.abs(result[1]:totable()[2] - 0.8427008) < 1e-5)
  assert_eq(result[2]:totable()[1], 1)
  assert_true(math.abs(result[2]:totable()[2] - 0.1572992) < 1e-5)
  local function bits(values)
    local text = {}
    for i, value in ipairs(values) do text[i] = tostring(value) end
    return table.concat(text, ",")
  end
  assert_eq(result[3]:dtype(), "bool")
  assert_eq(bits(result[3]:totable()), "false,false,true,false,false")
  assert_eq(bits(result[4]:totable()), "false,false,false,true,true")
  assert_eq(bits(result[5]:totable()), "true,true,false,false,false")
  session:close()
end)

test("builds zero/one tensors and stops gradients", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 2, 2 }, "like_x")
  local zeros = tf.zeros_like(graph, x, "like_zeros")
  local ones = tf.ones_like(graph, x, "like_ones")
  local same = tf.identity(graph, x, "like_identity")
  local stopped = tf.stop_gradient(graph, x, "like_stopped")
  local float64 = tf.placeholder(graph, "float64", { 2 }, "like_float64")
  local zeros64 = tf.zeros_like(graph, float64, "like_zeros64")
  local session = tf.Session.new(graph)
  local result = session:run({
    [x] = { { 1, 2 }, { 3, 4 } },
    [float64] = { 5, 6 },
  }, { zeros, ones, same, stopped, zeros64 })
  assert_eq(table.concat(result[1]:shape(), ","), "2,2")
  assert_eq(table.concat(result[1]:totable()[1], ","), "0,0")
  assert_eq(table.concat(result[2]:totable()[2], ","), "1,1")
  assert_eq(result[3]:totable()[2][1], 3)
  assert_eq(result[4]:totable()[1][2], 2)
  assert_eq(result[5]:dtype(), "float64")
  assert_eq(table.concat(result[5]:totable(), ","), "0,0")
  session:close()
end)

test("builds diagonal matrices and extracts diagonals", function()
  local graph = tf.Graph.new()
  local vector = tf.placeholder(graph, "float32", { 3 }, "diag_x")
  local matrix = tf.placeholder(graph, "float32", { 3, 3 }, "diag_m")
  local diagonal = tf.diag(graph, vector, "diag")
  local extracted = tf.diag_part(graph, matrix, "diag_part")
  local session = tf.Session.new(graph)
  local result = session:run({
    [vector] = { 1, 2, 3 },
    [matrix] = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
  }, { diagonal, extracted })
  assert_eq(table.concat(result[1]:shape(), ","), "3,3")
  assert_eq(result[1]:totable()[1][1], 1)
  assert_eq(result[1]:totable()[2][1], 0)
  assert_eq(result[1]:totable()[2][2], 2)
  assert_eq(result[1]:totable()[3][3], 3)
  assert_eq(table.concat(result[2]:shape(), ","), "3")
  assert_eq(table.concat(result[2]:totable(), ","), "1,5,9")
  session:close()
end)

test("builds batched diagonal matrices", function()
  local graph = tf.Graph.new()
  local batched = tf.placeholder(graph, "float32", { 2, 3 }, "mdiag_x")
  local matrix = tf.matrix_diag(graph, batched, "mdiag")
  local diagonal = tf.matrix_diag_part(graph, matrix, "mdiag_part")
  local session = tf.Session.new(graph)
  local result = session:run({ [batched] = { { 1, 2, 3 }, { 4, 5, 6 } } },
    { matrix, diagonal })
  assert_eq(table.concat(result[1]:shape(), ","), "2,3,3")
  assert_eq(result[1]:totable()[1][1][1], 1)
  assert_eq(result[1]:totable()[1][2][2], 2)
  assert_eq(result[1]:totable()[2][3][3], 6)
  assert_eq(table.concat(result[2]:shape(), ","), "2,3")
  assert_eq(table.concat(result[2]:totable()[2], ","), "4,5,6")
  session:close()
end)

test("masks matrix bands and replaces diagonals", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 3, 3 }, "band_x")
  local band = tf.matrix_band_part(graph, matrix, -1, 0, "band")
  local diagonal = tf.placeholder(graph, "float32", { 3 }, "band_diag")
  local replaced = tf.matrix_set_diag(graph, matrix, diagonal, "band_set")
  local session = tf.Session.new(graph)
  local result = session:run({
    [matrix] = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
    [diagonal] = { 9, 8, 7 },
  }, { band, replaced })
  assert_eq(table.concat(result[1]:totable()[2], ","), "4,5,0")
  assert_eq(table.concat(result[1]:totable()[3], ","), "7,8,9")
  assert_eq(table.concat(result[2]:totable()[1], ","), "9,2,3")
  assert_eq(table.concat(result[2]:totable()[2], ","), "4,8,6")
  assert_eq(table.concat(result[2]:totable()[3], ","), "7,8,7")
  assert_error(function()
    tf.matrix_band_part(graph, matrix, -2, 0, "band_bad")
  end, "num_lower")
  assert_error(function()
    tf.matrix_band_part(graph, matrix, -1, 0.5, "band_bad_upper")
  end, "num_upper")
  session:close()
end)

test("inverts, solves, and takes determinants of matrices", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 2, 2 }, "lin_x")
  local inverse = tf.matrix_inverse(graph, matrix, "lin_inverse")
  local determinant = tf.matrix_determinant(graph, matrix, "lin_det")
  local solve_matrix = tf.placeholder(graph, "float32", { 2, 2 }, "solve_m")
  local rhs = tf.placeholder(graph, "float32", { 2, 1 }, "solve_rhs")
  local solution = tf.matrix_solve(graph, solve_matrix, rhs, "solve")
  local session = tf.Session.new(graph)
  local result = session:run({
    [matrix] = { { 4, 7 }, { 2, 6 } },
    [solve_matrix] = { { 2, 0 }, { 0, 4 } },
    [rhs] = { { 2 }, { 8 } },
  }, { inverse, determinant, solution })
  assert_true(math.abs(result[1]:totable()[1][1] - 0.6) < 1e-5)
  assert_true(math.abs(result[1]:totable()[1][2] + 0.7) < 1e-5)
  assert_true(math.abs(result[1]:totable()[2][1] + 0.2) < 1e-5)
  assert_true(math.abs(result[1]:totable()[2][2] - 0.4) < 1e-5)
  assert_true(math.abs(result[2]:totable() - 10) < 1e-5)
  assert_true(math.abs(result[3]:totable()[1][1] - 1) < 1e-5)
  assert_true(math.abs(result[3]:totable()[2][1] - 2) < 1e-5)
  session:close()
end)

test("converts between depth and spatial blocks", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 4 }, "space_x")
  local expanded = tf.depth_to_space(graph, input, 2, "space_expand")
  local restored = tf.space_to_depth(graph, expanded, 2, "space_restore")
  local session = tf.Session.new(graph)
  local source = {
    { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
      { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } },
  }
  local result = session:run({ [input] = source }, { expanded, restored })
  assert_eq(table.concat(result[1]:shape(), ","), "1,4,4,1")
  assert_eq(table.concat(result[2]:shape(), ","), "1,2,2,4")
  local function flatten(value, output)
    output = output or {}
    if type(value) == "table" then
      for _, item in ipairs(value) do flatten(item, output) end
    else
      output[#output + 1] = value
    end
    return output
  end
  assert_eq(table.concat(flatten(result[2]:totable()), ","),
    table.concat(flatten(source), ","))
  assert_error(function()
    tf.depth_to_space(graph, input, 0, "space_bad_block")
  end, "positive")
  assert_error(function()
    tf.space_to_depth(graph, input, 2, "space_bad_format", nil, "XYZ")
  end, "NHWC or NCHW")
  session:close()
end)

test("moves blocks between batch and space", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 1 }, "sbn_x")
  local blocked = tf.space_to_batch_nd(graph, input, { 2, 2 },
    { { 0, 0 }, { 0, 0 } }, "sbn_to_batch")
  local restored = tf.batch_to_space_nd(graph, blocked, { 2, 2 },
    { { 0, 0 }, { 0, 0 } }, "sbn_to_space")
  local session = tf.Session.new(graph)
  local source = { { { { 1 }, { 2 } }, { { 3 }, { 4 } } } }
  local result = session:run({ [input] = source }, { blocked, restored })
  assert_eq(table.concat(result[1]:shape(), ","), "4,1,1,1")
  assert_eq(table.concat(result[2]:shape(), ","), "1,2,2,1")
  local function flatten(value, output)
    output = output or {}
    if type(value) == "table" then
      for _, item in ipairs(value) do flatten(item, output) end
    else
      output[#output + 1] = value
    end
    return output
  end
  assert_eq(table.concat(flatten(result[2]:totable()), ","),
    table.concat(flatten(source), ","))
  assert_error(function()
    tf.space_to_batch_nd(graph, input, { 2, 2 }, { { 0, 0 } }, "sbn_bad")
  end, "length")
  assert_error(function()
    tf.batch_to_space_nd(graph, blocked, { 0, 2 }, { { 0, 0 }, { 0, 0 } },
      "sbn_bad_block")
  end, "positive")
  session:close()
end)

test("clips values to a range", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "float32", { 3 }, "clip_x")
  local clipped = tf.clip_by_value(graph, x, -1, 1, "clip")
  local session = tf.Session.new(graph)
  local result = session:run({ [x] = { -2, 0, 3 } }, { clipped })[1]:totable()
  assert_eq(result[1], -1)
  assert_eq(result[2], 0)
  assert_eq(result[3], 1)
  session:close()
end)

test("transposes, expands, squeezes, and gathers graph outputs", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 2, 3 }, "rank_matrix")
  local transposed = tf.transpose(graph, matrix, { 1, 0 }, "rank_transpose")
  local expanded = tf.expand_dims(graph, matrix, 0, "rank_expand")
  local squeezed = tf.squeeze(graph, expanded, 0, "rank_squeeze")
  local values = tf.placeholder(graph, "float32", { 4 }, "rank_values")
  local indices = tf.placeholder(graph, "int32", { 2 }, "rank_indices")
  local gathered = tf.gather(graph, values, indices, 0, "rank_gather")
  local session = tf.Session.new(graph)
  local result = session:run({
    [matrix] = { { 1, 2, 3 }, { 4, 5, 6 } },
    [values] = { 10, 20, 30, 40 },
    [indices] = { 3, 1 },
  }, { transposed, expanded, squeezed, gathered })
  assert_eq(table.concat(result[1]:shape(), ","), "3,2")
  assert_eq(result[1]:totable()[1][2], 4)
  assert_eq(table.concat(result[2]:shape(), ","), "1,2,3")
  assert_eq(table.concat(result[3]:shape(), ","), "2,3")
  assert_eq(result[3]:totable()[2][3], 6)
  assert_eq(table.concat(result[4]:shape(), ","), "2")
  assert_eq(result[4]:totable()[1], 40)
  assert_eq(result[4]:totable()[2], 20)
  session:close()
end)

test("gathers and scatters with ND indices", function()
  local graph = tf.Graph.new()
  local params = tf.placeholder(graph, "float32", { 3, 2 }, "nd_params")
  local gather_indices = tf.placeholder(graph, "int32", { 2, 1 }, "nd_gi")
  local gathered = tf.gather_nd(graph, params, gather_indices, "nd_gather")
  local scatter_indices = tf.placeholder(graph, "int32", { 2, 1 }, "nd_si")
  local updates = tf.placeholder(graph, "float32", { 2, 2 }, "nd_updates")
  local scattered = tf.scatter_nd(graph, scatter_indices, updates, { 3, 2 },
    "nd_scatter")
  local session = tf.Session.new(graph)
  local result = session:run({
    [params] = { { 10, 11 }, { 20, 21 }, { 30, 31 } },
    [gather_indices] = { { 0 }, { 2 } },
    [scatter_indices] = { { 0 }, { 2 } },
    [updates] = { { 1, 2 }, { 3, 4 } },
  }, { gathered, scattered })
  assert_eq(table.concat(result[1]:shape(), ","), "2,2")
  assert_eq(result[1]:totable()[1][1], 10)
  assert_eq(result[1]:totable()[2][2], 31)
  assert_eq(table.concat(result[2]:shape(), ","), "3,2")
  assert_eq(result[2]:totable()[1][2], 2)
  assert_eq(result[2]:totable()[2][1], 0)
  assert_eq(result[2]:totable()[3][1], 3)
  assert_error(function()
    tf.scatter_nd(graph, scatter_indices, updates, {}, "nd_bad_shape")
  end, "non-empty")
  assert_error(function()
    tf.scatter_nd(graph, scatter_indices, updates, { -1, 2 }, "nd_negative")
  end, "non-negative")
  session:close()
end)

test("scatters update, add, sub, min, and max with ND indices", function()
  local graph = tf.Graph.new()
  local tensor = tf.placeholder(graph, "float32", { 3, 2 }, "ts_tensor")
  local indices = tf.placeholder(graph, "int32", { 1, 1 }, "ts_indices")
  local updates = tf.placeholder(graph, "float32", { 1, 2 }, "ts_updates")
  local updated = tf.tensor_scatter_nd_update(graph, tensor, indices, updates,
    "ts_update")
  local added = tf.tensor_scatter_nd_add(graph, tensor, indices, updates,
    "ts_add")
  local subtracted = tf.tensor_scatter_nd_sub(graph, tensor, indices, updates,
    "ts_sub")
  local minimal = tf.tensor_scatter_nd_min(graph, tensor, indices, updates,
    "ts_min")
  local maximal = tf.tensor_scatter_nd_max(graph, tensor, indices, updates,
    "ts_max")
  local session = tf.Session.new(graph)
  local result = session:run({
    [tensor] = { { 1, 2 }, { 3, 4 }, { 5, 6 } },
    [indices] = { { 1 } },
    [updates] = { { 9, 9 } },
  }, { updated, added, subtracted, minimal, maximal })
  assert_eq(table.concat(result[1]:shape(), ","), "3,2")
  assert_eq(table.concat(result[1]:totable()[2], ","), "9,9")
  assert_eq(table.concat(result[1]:totable()[1], ","), "1,2")
  assert_eq(table.concat(result[2]:totable()[2], ","), "12,13")
  assert_eq(table.concat(result[2]:totable()[3], ","), "5,6")
  assert_eq(table.concat(result[3]:totable()[2], ","), "-6,-5")
  assert_eq(table.concat(result[4]:totable()[2], ","), "3,4")
  assert_eq(table.concat(result[5]:totable()[2], ","), "9,9")
  session:close()
end)

test("finds unique values, indices, and counts", function()
  local graph = tf.Graph.new()
  local value = tf.placeholder(graph, "int32", { 4 }, "unique_x")
  local unique = tf.unique(graph, value, "unique")
  local counted = tf.unique_with_counts(graph, value, "unique_counts")
  local session = tf.Session.new(graph)
  local result = session:run({ [value] = { 3, 1, 3, 2 } },
    { unique[1], unique[2], counted[1], counted[2], counted[3] })
  assert_eq(table.concat(result[1]:totable(), ","), "3,1,2")
  assert_eq(table.concat(result[2]:totable(), ","), "0,1,0,2")
  assert_eq(table.concat(result[3]:totable(), ","), "3,1,2")
  assert_eq(table.concat(result[4]:totable(), ","), "0,1,0,2")
  assert_eq(table.concat(result[5]:totable(), ","), "2,1,1")
  assert_eq(result[1]:dtype(), "int32")
  assert_eq(result[2]:dtype(), "int32")
  assert_eq(result[5]:dtype(), "int32")
  session:close()
end)

test("masks tensor rows with a boolean vector", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 2, 3 }, "mask_x")
  local mask = tf.placeholder(graph, "bool", { 2 }, "mask_b")
  local masked = tf.boolean_mask(graph, matrix, mask, "mask")
  local session = tf.Session.new(graph)
  local result = session:run({
    [matrix] = { { 1, 2, 3 }, { 4, 5, 6 } },
    [mask] = { true, false },
  }, { masked })[1]
  assert_eq(table.concat(result:shape(), ","), "1,3")
  assert_eq(table.concat(result:totable()[1], ","), "1,2,3")
  session:close()
end)

test("builds sequence masks from lengths", function()
  local graph = tf.Graph.new()
  local lengths = tf.placeholder(graph, "int32", { 3 }, "sm_lengths")
  local mask = tf.sequence_mask(graph, lengths, 4, "sm_mask")
  local session = tf.Session.new(graph)
  local result = session:run({ [lengths] = { 2, 0, 3 } }, { mask })[1]
  assert_eq(table.concat(result:shape(), ","), "3,4")
  assert_eq(table.concat(result:totable()[1], ","), "1,1,0,0")
  assert_eq(table.concat(result:totable()[2], ","), "0,0,0,0")
  assert_eq(table.concat(result:totable()[3], ","), "1,1,1,0")
  assert_error(function()
    tf.sequence_mask(graph, lengths, 0, "sm_bad")
  end, "positive")
  session:close()
end)

test("splits tensors into equal and explicit parts", function()
  local graph = tf.Graph.new()
  local value = tf.placeholder(graph, "float32", { 6 }, "split_value")
  local equal = tf.split(graph, value, 3, 0, "split_equal")
  local uneven = tf.split(graph, value, { 1, 2, 3 }, 0, "split_sizes")
  assert_eq(#equal, 3)
  assert_eq(#uneven, 3)
  local session = tf.Session.new(graph)
  local parts = session:run({ [value] = { 1, 2, 3, 4, 5, 6 } },
    { equal[1], equal[2], equal[3], uneven[1], uneven[2], uneven[3] })
  assert_eq(table.concat(parts[1]:shape(), ","), "2")
  assert_eq(parts[1]:totable()[1], 1)
  assert_eq(parts[2]:totable()[2], 4)
  assert_eq(parts[3]:totable()[1], 5)
  assert_eq(table.concat(parts[4]:shape(), ","), "1")
  assert_eq(table.concat(parts[5]:shape(), ","), "2")
  assert_eq(table.concat(parts[6]:shape(), ","), "3")
  assert_eq(parts[6]:totable()[3], 6)
  assert_error(function() tf.split(graph, value, 0, 0, "bad_count") end,
    "positive")
  assert_error(function() tf.split(graph, value, { 1, -1 }, 0, "bad_size") end,
    "non-negative")
  assert_error(function() tf.split(graph, value, 1.5, 0, "bad_split") end,
    "split")
  session:close()
end)

test("pads, tiles, and selects graph tensors", function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 2, 2 }, "pad_matrix")
  local padded = tf.pad(graph, matrix, { { 1, 1 }, { 0, 1 } }, "CONSTANT",
    "pad_constant")
  local reflected = tf.pad(graph, matrix, { { 1, 1 }, { 1, 1 } }, "REFLECT",
    "pad_reflect")
  local vector = tf.placeholder(graph, "float32", { 2 }, "tile_vector")
  local tiled = tf.tile(graph, vector, { 3 }, "tile_value")
  local condition = tf.placeholder(graph, "bool", { 3 }, "where_condition")
  local choices = tf.placeholder(graph, "float32", { 3 }, "where_choices")
  local other = tf.constant(graph, { 10, 20, 30 }, "where_other")
  local selected = tf.where(graph, condition, choices, other, "where_value")

  local session = tf.Session.new(graph)
  local result = session:run({
    [matrix] = { { 1, 2 }, { 3, 4 } },
    [vector] = { 5, 6 },
    [condition] = { true, false, true },
    [choices] = { 1, 2, 3 },
  }, { padded, reflected, tiled, selected })
  assert_eq(table.concat(result[1]:shape(), ","), "4,3")
  assert_eq(result[1]:totable()[1][1], 0)
  assert_eq(result[1]:totable()[2][1], 1)
  assert_eq(result[1]:totable()[2][3], 0)
  assert_eq(result[1]:totable()[3][2], 4)
  assert_eq(table.concat(result[2]:shape(), ","), "4,4")
  assert_eq(table.concat(result[3]:shape(), ","), "6")
  assert_eq(result[3]:totable()[3], 5)
  assert_eq(result[3]:totable()[4], 6)
  assert_eq(result[4]:totable()[1], 1)
  assert_eq(result[4]:totable()[2], 20)
  assert_eq(result[4]:totable()[3], 3)

  assert_error(function()
    tf.pad(graph, matrix, { { 1 } }, "CONSTANT", "bad_pad")
  end, "pairs")
  assert_error(function()
    tf.pad(graph, matrix, { { 1, 1 }, { 1, 1 } }, "WRAP", "bad_mode")
  end, "mode")
  assert_error(function() tf.tile(graph, vector, {}, "bad_tile") end,
    "multiples")
  assert_error(function() tf.tile(graph, vector, { 1.5 }, "bad_multiple") end,
    "integers")
  session:close()
end)

test("broadcasts tensors and rolls elements", function()
  local graph = tf.Graph.new()
  local vector = tf.placeholder(graph, "float32", { 2 }, "br_x")
  local broadcast = tf.broadcast_to(graph, vector, { 3, 2 }, "br")
  local sequence = tf.placeholder(graph, "int32", { 4 }, "roll_x")
  local rolled = tf.roll(graph, sequence, 1, 0, "roll")
  local session = tf.Session.new(graph)
  local result = session:run({
    [vector] = { 7, 8 },
    [sequence] = { 1, 2, 3, 4 },
  }, { broadcast, rolled })
  assert_eq(table.concat(result[1]:shape(), ","), "3,2")
  assert_eq(table.concat(result[1]:totable()[2], ","), "7,8")
  assert_eq(result[1]:totable()[3][2], 8)
  assert_eq(table.concat(result[2]:totable(), ","), "4,1,2,3")
  assert_error(function()
    tf.broadcast_to(graph, vector, {}, "br_bad")
  end, "non-empty")
  assert_error(function()
    tf.roll(graph, sequence, { 1, 2 }, 0, "roll_bad")
  end, "match")
  session:close()
end)

test("squeezes all singleton dimensions and validates axes", function()
  local graph = tf.Graph.new()
  local value = tf.constant(graph, { { { 1 }, { 2 }, { 3 } } }, "squeeze_value",
    "float32")
  local squeezed = tf.squeeze(graph, value, nil, "squeeze_all")
  local session = tf.Session.new(graph)
  local result = session:run({}, { squeezed })[1]
  assert_eq(table.concat(result:shape(), ","), "3")
  assert_eq(result:totable()[3], 3)
  assert_error(function() tf.squeeze(graph, value, 0.5, "bad_squeeze") end,
    "axes")
  assert_error(function() tf.transpose(graph, value, nil, "bad_transpose") end,
    "permutation")
  session:close()
end)

test("stacks, unstacks, slices, and reverses graph tensors", function()
  local graph = tf.Graph.new()
  local a = tf.placeholder(graph, "float32", { 2 }, "st_a")
  local b = tf.placeholder(graph, "float32", { 2 }, "st_b")
  local rows = tf.stack(graph, { a, b }, 0, "st_rows")
  local columns = tf.stack(graph, { a, b }, 1, "st_columns")
  local matrix = tf.placeholder(graph, "float32", { 2, 3 }, "st_matrix")
  local row_parts = tf.unstack(graph, matrix, 2, 0, "st_row_parts")
  local column_parts = tf.unstack(graph, matrix, 3, 1, "st_column_parts")
  local sliced = tf.slice(graph, matrix, { 1, 1 }, { 1, 2 }, "st_slice")
  local vector = tf.placeholder(graph, "float32", { 4 }, "st_vector")
  local reversed = tf.reverse(graph, vector, { 0 }, "st_reverse")
  local row_reversed = tf.reverse(graph, matrix, { 1 }, "st_row_reverse")

  local session = tf.Session.new(graph)
  local result = session:run({
    [a] = { 1, 2 },
    [b] = { 10, 20 },
    [matrix] = { { 1, 2, 3 }, { 4, 5, 6 } },
    [vector] = { 1, 2, 3, 4 },
  }, { rows, columns, row_parts[1], row_parts[2], column_parts[1],
    column_parts[2], column_parts[3], sliced, reversed, row_reversed })

  assert_eq(table.concat(result[1]:shape(), ","), "2,2")
  assert_eq(result[1]:totable()[1][1], 1)
  assert_eq(result[1]:totable()[2][2], 20)
  assert_eq(table.concat(result[2]:shape(), ","), "2,2")
  assert_eq(result[2]:totable()[1][1], 1)
  assert_eq(result[2]:totable()[1][2], 10)
  assert_eq(result[2]:totable()[2][1], 2)
  assert_eq(table.concat(result[3]:shape(), ","), "3")
  assert_eq(result[3]:totable()[1], 1)
  assert_eq(result[4]:totable()[3], 6)
  assert_eq(table.concat(result[5]:shape(), ","), "2")
  assert_eq(result[5]:totable()[1], 1)
  assert_eq(result[5]:totable()[2], 4)
  assert_eq(result[6]:totable()[1], 2)
  assert_eq(result[6]:totable()[2], 5)
  assert_eq(result[7]:totable()[2], 6)
  assert_eq(table.concat(result[8]:shape(), ","), "1,2")
  assert_eq(result[8]:totable()[1][1], 5)
  assert_eq(result[8]:totable()[1][2], 6)
  assert_eq(table.concat(result[9]:totable(), ","), "4,3,2,1")
  assert_eq(result[10]:totable()[1][1], 3)
  assert_eq(result[10]:totable()[2][3], 4)

  assert_error(function() tf.stack(graph, {}, 0, "bad_stack_empty") end,
    "non-empty")
  assert_error(function() tf.stack(graph, { a }, 0.5, "bad_stack_axis") end,
    "axis")
  assert_error(function() tf.unstack(graph, matrix, 0, 0, "bad_unstack") end,
    "positive")
  assert_error(function()
    tf.slice(graph, matrix, {}, { 1, 1 }, "bad_slice_begin")
  end, "matching")
  assert_error(function()
    tf.slice(graph, matrix, { 0, 0 }, { 1 }, "bad_slice_size")
  end, "matching")
  assert_error(function()
    tf.slice(graph, matrix, { 0, 0 }, { -2, 1 }, "bad_slice_negative")
  end, "greater")
  assert_error(function() tf.reverse(graph, vector, {}, "bad_reverse") end,
    "axis")
  assert_error(function()
    tf.reverse(graph, vector, { 0.5 }, "bad_reverse_axis")
  end, "integers")
  session:close()
end)

test("inverts permutations and reverses sequences", function()
  local graph = tf.Graph.new()
  local permutation = tf.placeholder(graph, "int32", { 3 }, "perm_x")
  local inverse = tf.invert_permutation(graph, permutation, "perm_inverse")
  local sequence = tf.placeholder(graph, "float32", { 2, 3, 2 }, "seq_x")
  local lengths = tf.placeholder(graph, "int32", { 2 }, "seq_lengths")
  local reversed = tf.reverse_sequence(graph, sequence, lengths, 1, 0,
    "seq_reverse")
  local session = tf.Session.new(graph)
  local result = session:run({
    [permutation] = { 2, 0, 1 },
    [sequence] = {
      { { 1, 2 }, { 3, 4 }, { 5, 6 } },
      { { 7, 8 }, { 9, 10 }, { 11, 12 } },
    },
    [lengths] = { 2, 3 },
  }, { inverse, reversed })
  assert_eq(result[1]:dtype(), "int32")
  assert_eq(table.concat(result[1]:totable(), ","), "1,2,0")
  assert_eq(table.concat(result[2]:shape(), ","), "2,3,2")
  assert_eq(table.concat(result[2]:totable()[1][1], ","), "3,4")
  assert_eq(table.concat(result[2]:totable()[1][2], ","), "1,2")
  assert_eq(table.concat(result[2]:totable()[2][1], ","), "11,12")
  assert_eq(table.concat(result[2]:totable()[2][3], ","), "7,8")
  assert_error(function()
    tf.reverse_sequence(graph, sequence, lengths, -1, 0, "bad_seq_dim")
  end, "non-negative")
  assert_error(function()
    tf.reverse_sequence(graph, sequence, lengths, 1, 0.5, "bad_batch_dim")
  end, "non-negative")
  session:close()
end)

test("compares values and evaluates logical graph ops", function()
  local graph = tf.Graph.new()
  local x = tf.placeholder(graph, "int32", { 3 }, "cmp_x")
  local y = tf.placeholder(graph, "int32", { 3 }, "cmp_y")
  local equal = tf.equal(graph, x, y, "cmp_equal")
  local not_equal = tf.not_equal(graph, x, y, "cmp_not_equal")
  local greater = tf.greater(graph, x, y, "cmp_greater")
  local greater_equal = tf.greater_equal(graph, x, y, "cmp_greater_equal")
  local less = tf.less(graph, x, y, "cmp_less")
  local less_equal = tf.less_equal(graph, x, y, "cmp_less_equal")
  local a = tf.placeholder(graph, "bool", { 3 }, "logic_a")
  local b = tf.placeholder(graph, "bool", { 3 }, "logic_b")
  local conjunction = tf.logical_and(graph, a, b, "logic_and")
  local disjunction = tf.logical_or(graph, a, b, "logic_or")
  local exclusive = tf.logical_xor(graph, a, b, "logic_xor")
  local negation = tf.logical_not(graph, a, "logic_not")
  local session = tf.Session.new(graph)
  local result = session:run({
    [x] = { 1, 2, 3 },
    [y] = { 2, 2, 2 },
    [a] = { true, true, false },
    [b] = { true, false, false },
  }, { equal, not_equal, greater, greater_equal, less, less_equal,
    conjunction, disjunction, exclusive, negation })

  local function bits(values)
    local text = {}
    for i, value in ipairs(values) do text[i] = tostring(value) end
    return table.concat(text, ",")
  end
  assert_eq(result[1]:dtype(), "bool")
  assert_eq(bits(result[1]:totable()), "false,true,false")
  assert_eq(bits(result[2]:totable()), "true,false,true")
  assert_eq(bits(result[3]:totable()), "false,false,true")
  assert_eq(bits(result[4]:totable()), "false,true,true")
  assert_eq(bits(result[5]:totable()), "true,false,false")
  assert_eq(bits(result[6]:totable()), "true,true,false")
  assert_eq(bits(result[7]:totable()), "true,false,false")
  assert_eq(bits(result[8]:totable()), "true,true,false")
  assert_eq(bits(result[9]:totable()), "false,true,false")
  assert_eq(bits(result[10]:totable()), "false,false,true")
  session:close()
end)

test("reduces products and boolean tensors and squares differences", function()
  local graph = tf.Graph.new()
  local values = tf.placeholder(graph, "float32", { 2, 3 }, "ext_values")
  local product = tf.reduce_prod(graph, values, { 1 }, "ext_product")
  local expected = tf.placeholder(graph, "float32", { 2, 3 }, "ext_expected")
  local difference = tf.squared_difference(graph, values, expected,
    "ext_difference")
  local flags = tf.placeholder(graph, "bool", { 2, 3 }, "ext_flags")
  local all_rows = tf.reduce_all(graph, flags, { 1 }, "ext_all_rows")
  local any_rows = tf.reduce_any(graph, flags, { 1 }, "ext_any_rows")
  local all_values = tf.reduce_all(graph, flags, { 0, 1 }, "ext_all_values")
  local session = tf.Session.new(graph)
  local result = session:run({
    [values] = { { 1, 2, 3 }, { 4, 5, 6 } },
    [expected] = { { 1, 1, 1 }, { 4, 4, 4 } },
    [flags] = { { true, true, false }, { true, true, true } },
  }, { product, difference, all_rows, any_rows, all_values })

  assert_eq(table.concat(result[1]:totable(), ","), "6,120")
  assert_eq(result[2]:totable()[1][3], 4)
  assert_eq(result[2]:totable()[2][3], 4)
  assert_eq(result[3]:dtype(), "bool")
  assert_eq(result[3]:totable()[1], false)
  assert_eq(result[3]:totable()[2], true)
  assert_eq(result[4]:totable()[1], true)
  assert_eq(result[4]:totable()[2], true)
  assert_eq(result[5]:totable(), false)
  session:close()
end)
