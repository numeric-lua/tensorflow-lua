local tf = require("tensorflow")

test("saves and loads a portable deployment bundle", function()
  local model = tf.Sequential.new({ input_shape = { 1 } })
  model:add({ units = 1 })
  model:compile({ optimizer = { type = "sgd", learning_rate = 0.01 },
    loss = "mse" })
  local x = tf.tensor({ { 1 }, { 2 }, { 3 } })
  local y = tf.tensor({ { 2 }, { 4 }, { 6 } })
  model:fit_tensor(x, y, { epochs = 2 })
  local directory = (TMPDIR or "/tmp") .. "/tensorflow-lua-bundle"
  os.execute("rm -rf '" .. directory .. "'")
  model:save_bundle(directory)
  local bundle = tf.load_bundle(directory)
  assert_true(bundle.manifest.format == "tensorflow-lua-bundle")
  local output = bundle.session:run({ [bundle.inputs[1]] = x },
    { bundle.outputs[1] })[1]
  assert_eq(output:shape()[1], 3)
  assert_true(output:numel() > 0)
  output:destroy()
  bundle.session:close()
  bundle.graph:delete()
  x:destroy(); y:destroy(); model:close()
end)
