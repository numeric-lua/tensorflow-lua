local tf = require("tensorflow")


test("Tensor:slice and take_rows preserve bytes and shapes", function()
  local tensor = tf.tensor({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  assert_eq(table.concat(tensor:slice(2, 3):shape(), ","), "2,2")
  assert_eq(tensor:slice(2, 3):totable()[1][1], 3)
  local selected = tensor:take_rows({ 3, 1 })
  assert_eq(table.concat(selected:shape(), ","), "2,2")
  assert_eq(selected:totable()[1][1], 5)
  assert_eq(selected:totable()[2][2], 2)
  tensor:destroy(); selected:destroy()
end)
