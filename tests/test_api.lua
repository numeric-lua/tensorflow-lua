local tf = require("tensorflow")

test("exposes the public API surface", function()
  local names = {
    "VERSION", "Tensor", "Graph", "Session", "CheckpointReader",
    "Sequential", "tensor", "complex", "placeholder", "constant", "variable",
    "gradients", "reshape", "cast", "concat", "clip_by_value", "reduce_mean",
    "reduce_sum", "reduce_max", "reduce_min", "argmax", "one_hot", "exp",
    "log", "sqrt", "save_v2", "restore_v2", "save_checkpoint",
    "load_checkpoint", "load_saved_model", "version", "train", "nn", "layers",
    "losses", "metrics", "callbacks", "data", "eager",
  }
  for _, name in ipairs(names) do
    assert_true(tf[name] ~= nil, "missing tf." .. name)
  end
  local train_functions = {
    "gradient_descent", "adam", "adamw", "rmsprop", "centered_rmsprop",
    "adagrad", "adadelta",
    "ftrl", "proximal_adagrad", "momentum", "Saver",
  }
  for _, name in ipairs(train_functions) do
    assert_true(tf.train[name] ~= nil, "missing tf.train." .. name)
  end
  local nn_functions = {
    "relu", "leaky_relu", "elu", "sigmoid", "tanh", "softmax", "softplus",
  }
  for _, name in ipairs(nn_functions) do
    assert_true(tf.nn[name] ~= nil, "missing tf.nn." .. name)
  end
  assert_true(tf.data.Dataset ~= nil)
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
  assert_true(tf.layers.dense ~= nil and tf.layers.embedding ~= nil)
  assert_true(tf.layers.conv2d ~= nil and tf.layers.max_pool2d ~= nil)
  assert_true(tf.layers.batch_normalization ~= nil)
  assert_true(tf.losses.mean_squared_error ~= nil)
  assert_true(tf.metrics.binary_accuracy ~= nil)
  assert_eq(tf.VERSION, "0.8.0")
end)
