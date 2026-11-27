local tf = require("tensorflow")

test("exposes the public API surface", function()
  local names = {
    "VERSION", "Tensor", "Graph", "Session", "CheckpointReader",
    "Sequential", "tensor", "complex", "placeholder", "constant", "variable",
    "gradients", "reshape", "transpose", "expand_dims", "squeeze", "split",
    "pad", "tile", "broadcast_to", "roll", "where", "stack", "unstack",
    "slice", "reverse", "gather",
    "invert_permutation", "reverse_sequence", "boolean_mask",
    "depth_to_space", "space_to_depth", "space_to_batch_nd",
    "batch_to_space_nd", "gather_nd", "scatter_nd",
    "tensor_scatter_nd_update", "tensor_scatter_nd_add",
    "tensor_scatter_nd_sub", "tensor_scatter_nd_min",
    "tensor_scatter_nd_max", "unique", "unique_with_counts", "sequence_mask",
    "cast", "concat", "clip_by_value", "add_n", "zeros_like", "ones_like",
    "identity", "stop_gradient", "real", "imag", "fft", "ifft", "diag",
    "diag_part", "matrix_diag",
    "matrix_diag_part", "matrix_band_part", "matrix_set_diag",
    "matrix_inverse", "matrix_determinant", "matrix_solve", "cholesky",
    "matrix_triangular_solve", "qr", "self_adjoint_eig", "svd",
    "batch_matmul", "fill", "zeros", "ones", "random_uniform",
    "random_normal", "random_shuffle", "cumsum", "cumprod",
    "range",
    "linspace", "reduce_mean",
    "reduce_sum", "reduce_logsumexp", "reduce_prod", "reduce_max",
    "reduce_min", "reduce_variance", "reduce_stddev", "norm", "l2_normalize",
    "reduce_all",
    "reduce_any", "argmax", "argmin", "top_k", "in_top_k", "one_hot",
    "bincount", "exp",
    "equal", "not_equal",
    "greater",
    "greater_equal", "less", "less_equal", "logical_and", "logical_or",
    "logical_xor", "logical_not", "squared_difference",
    "log", "sqrt", "erf", "erfc", "is_nan", "is_inf", "is_finite", "floor",
    "ceil", "round", "sign", "rsqrt", "reciprocal",
    "save_v2", "restore_v2", "save_checkpoint", "checkpoint_metadata",
    "load_checkpoint", "load_saved_model", "version", "train", "nn", "layers",
    "losses", "metrics", "callbacks", "data", "eager", "schedules",
    "Functional",
  }
  for _, name in ipairs(names) do
    assert_true(tf[name] ~= nil, "missing tf." .. name)
  end
  local train_functions = {
    "gradient_descent", "sgd", "optimizer", "adam", "adamw", "adamax",
    "rmsprop", "centered_rmsprop",
    "adagrad", "adadelta", "clip_by_norm", "clip_by_global_norm",
    "ftrl", "proximal_adagrad", "momentum", "exponential_moving_average",
    "gradient_accumulation", "apply_gradients", "Saver",
  }
  for _, name in ipairs(train_functions) do
    assert_true(tf.train[name] ~= nil, "missing tf.train." .. name)
  end
  local nn_functions = {
    "relu", "relu6", "leaky_relu", "elu", "softsign", "selu", "sigmoid",
    "tanh", "softmax", "log_softmax", "softplus",
  }
  for _, name in ipairs(nn_functions) do
    assert_true(tf.nn[name] ~= nil, "missing tf.nn." .. name)
  end
  assert_true(tf.data.Dataset ~= nil)
  assert_true(tf.data.Dataset.from_tensor ~= nil)
  assert_true(tf.save_bundle ~= nil and tf.load_bundle ~= nil)
  assert_true(tf.callbacks.EarlyStopping ~= nil)
  assert_true(tf.callbacks.ModelCheckpoint ~= nil)
  local sequential_methods = {
    "new", "add", "compile", "fit", "fit_dataset", "evaluate",
    "evaluate_dataset", "predict", "predict_dataset", "predict_classes",
    "predict_proba", "get_weights", "set_weights", "count_params", "summary",
    "save", "load", "save_native", "load_native", "export", "close",
  }
  for _, name in ipairs(sequential_methods) do
    assert_true(type(tf.Sequential[name]) == "function",
      "missing Sequential." .. name)
  end
  assert_true(tf.eager.Context ~= nil)
  local eager_methods = {
    "stack", "unstack", "one_hot", "clip_by_value", "equal", "not_equal",
    "greater", "greater_equal", "less", "less_equal", "logical_and",
    "logical_or", "logical_xor", "logical_not", "reduce_all", "reduce_any",
    "reduce_logsumexp", "argmin", "relu6", "add_n", "batch_matmul", "fill",
    "range", "linspace", "log_softmax", "floor", "ceil", "round", "sign",
    "rsqrt", "reciprocal", "zeros_like", "ones_like", "identity",
    "stop_gradient", "diag", "diag_part", "invert_permutation",
    "reverse_sequence", "softsign", "selu", "gather_nd", "scatter_nd",
    "tensor_scatter_nd_update", "tensor_scatter_nd_add",
    "tensor_scatter_nd_sub", "tensor_scatter_nd_min", "tensor_scatter_nd_max",
    "sequence_mask", "top_k", "in_top_k", "space_to_batch_nd",
    "batch_to_space_nd", "unique", "unique_with_counts", "erf",
    "erfc", "is_nan", "is_inf", "is_finite",
    "reduce_variance", "reduce_stddev", "norm", "l2_normalize", "matrix_diag",
    "matrix_diag_part", "matrix_band_part", "matrix_set_diag",
    "matrix_inverse", "matrix_determinant", "matrix_solve", "cholesky",
    "matrix_triangular_solve", "qr", "self_adjoint_eig", "svd",
    "random_uniform", "random_normal", "bincount", "cumsum", "cumprod",
    "real", "imag", "fft", "ifft", "zeros", "ones", "broadcast_to",
    "roll", "maximum", "minimum",
  }
  for _, name in ipairs(eager_methods) do
    assert_true(type(tf.eager.Context[name]) == "function",
      "missing eager Context." .. name)
  end
  assert_true(tf.layers.dense ~= nil and tf.layers.embedding ~= nil)
  assert_true(tf.layers.conv2d ~= nil and tf.layers.max_pool2d ~= nil)
  assert_true(tf.layers.conv2d_transpose ~= nil)
  assert_true(tf.layers.max_pool1d ~= nil and tf.layers.avg_pool1d ~= nil)
  assert_true(tf.layers.global_avg_pool1d ~= nil)
  assert_true(tf.layers.separable_conv2d ~= nil)
  assert_true(tf.layers.layer_normalization ~= nil)
  assert_true(tf.layers.conv3d ~= nil and tf.layers.max_pool3d ~= nil)
  assert_true(tf.layers.global_avg_pool3d ~= nil)
  assert_true(tf.layers.upsampling3d ~= nil)
  assert_true(tf.layers.lstm ~= nil and tf.layers.gru ~= nil)
  assert_true(tf.layers.attention ~= nil)
  assert_true(tf.layers.multi_head_attention ~= nil)
  assert_true(tf.layers.transformer_block ~= nil)
  assert_true(tf.layers.positional_encoding ~= nil)
  assert_true(tf.layers.batch_normalization ~= nil)
  assert_true(tf.losses.mean_squared_error ~= nil)
  assert_true(tf.losses.huber ~= nil and tf.losses.hinge ~= nil)
  assert_true(tf.losses.categorical_hinge ~= nil)
  assert_true(tf.metrics.binary_accuracy ~= nil)
  assert_true(tf.metrics.precision ~= nil and tf.metrics.recall ~= nil)
  assert_true(tf.metrics.f1 ~= nil)
  assert_true(tf.metrics.auc ~= nil and tf.metrics.multiclass_prf ~= nil)
  assert_eq(tf.VERSION, "2.8.0")
end)
