--[[

  Fast Binary Tree LSTM
  Two techniques are usd to speed up tree LSTM:
  1. Following the implementation of FastLSTM in rnn, all gates in a node are calculated together.
  2. The LSTM node allocated at differeent tree nodes (in different trees) are cached to avoid the allocation time.

--]]

require("torch")
require("cutorch")
require("nn")
require("cunn")
require("nngraph")
require("rnn")


require("utils")

torch.class("BinaryTreeLSTM", "nn.Module")

function BinaryTreeLSTM:__init(config)
  self.input_dim = config.input_dim
  self.output_dim = config.output_dim
  self.module_name = config.name
  self.output_name = self.module_name .. "_output"
  self.grad_output_name = self.output_name .. "_grad"
  self.get_input = config.get_input
  self.acc_grad_input = config.acc_grad_input

  self.empty_output = torch.zeros(self.output_dim):cuda()
  self.empty_input = torch.zeros(self.input_dim):cuda()

  -- create shared modules for leaf and composer
  self.modules = {self:new_module():cuda()}

end


function BinaryTreeLSTM:get_module(module_id)
  if #self.modules < module_id then
    local new = self:new_module():cuda()
    share_params(new, self.modules[1])
    self.modules[#self.modules + 1] = new
    return self:get_module(module_id)
  else
    return self.modules[module_id]
  end
end


function BinaryTreeLSTM:free_modules()
  for i = 2, #self.modules do self.modules[i] = nil end
end


function BinaryTreeLSTM:training()
  self.train = true
end


function BinaryTreeLSTM:evaluate()
  self.train = false
end


----------------------------------------------------
-- define the network
----------------------------------------------------

function BinaryTreeLSTM:new_module()
  -- calc the 4 gates at one step
  -- input {x, lh, rh, lc, rc},
  -- output {h, c}

  local x = nn.Identity()()
  local lh = nn.Identity()()
  local rh = nn.Identity()()
  local lc = nn.Identity()()
  local rc = nn.Identity()()

  local i2g = nn.Linear(self.input_dim, 5*self.output_dim)(x)
  local lo2g = nn.LinearNoBias(self.output_dim, 5*self.output_dim)(lh)
  local ro2g = nn.LinearNoBias(self.output_dim, 5*self.output_dim)(rh)

  local sums = nn.CAddTable(){i2g, lo2g, ro2g}

  local sigmoid_chunk = nn.Sigmoid()(nn.Narrow(1, 1, 4*self.output_dim)(sums))

  local input_gate = nn.Narrow(1, 1, self.output_dim)(sigmoid_chunk)
  local lf_gate = nn.Narrow(1, self.output_dim+1, self.output_dim)(sigmoid_chunk)
  local rf_gate = nn.Narrow(1, 2*self.output_dim+1, self.output_dim)(sigmoid_chunk)
  local output_gate = nn.Narrow(1, 3*self.output_dim+1, self.output_dim)(sigmoid_chunk)

  local hidden = nn.Tanh()(nn.Narrow(1, 4*self.output_dim, self.output_dim)(sums))

  local c = nn.CAddTable(){
    nn.CMulTable(){input_gate, hidden},
    nn.CMulTable(){lf_gate, lc},
    nn.CMulTable(){rf_gate, rc}
                          }

  local h = nn.CMulTable(){output_gate, nn.Tanh()(c)}

  return nn.gModule({x, lh, rh, lc, rc}, {h, c})

end


----------------------------------------------------
-- set up forward and backward
----------------------------------------------------


function BinaryTreeLSTM:forward(tree, inputs, offset)
  return self:_forward(tree, inputs, offset or 0)[1]
end


function BinaryTreeLSTM:_forward(tree, inputs, module_offset)
  local input = self.get_input(inputs, tree) or self.empty_input
  local lh, rh, lc, rc
  if tree.val ~= nil then
    lh, lc = self.empty_output, self.empty_output
    rh, rc = self.empty_output, self.empty_output
  else
    local lvecs = self:_forward(tree.children[1], inputs, module_offset)
    local rvecs = self:_forward(tree.children[2], inputs, module_offset)

    lh, lc, rh, rc = self:get_children_outputs(tree)
  end

  tree[self.module_name] = self:get_module(tree.postorder_id + 2*module_offset)
  tree[self.output_name] = tree[self.module_name]:forward{input, lh, rh, lc, rc}

  return tree[self.output_name]
end


function BinaryTreeLSTM:backward(tree, inputs, grad_inputs)
  self:_backward(tree, inputs, grad_inputs)
end


function BinaryTreeLSTM:_backward(tree, inputs, grad_inputs)
  local input = self.get_input(inputs, tree) or self.empty_input
  local lh, lc, rh, rc

  if tree.val ~= nil then
    lh, lc = self.empty_output, self.empty_output
    rh, rc = self.empty_output, self.empty_output
  else
    lh, lc, rh, rc = self:get_children_outputs(tree)
  end

  local grad = tree[self.module_name]:backward(
    {input, lh, rh, lc, rc},
    tree[self.grad_output_name])

  self.acc_grad_input(grad_inputs, tree, grad[1])

  if tree.val == nil then
    self:acc_grad_output(tree.children[1], {grad[2], grad[4]})
    self:acc_grad_output(tree.children[2], {grad[3], grad[5]})

    self:_backward(tree.children[1], inputs, grad_inputs)
    self:_backward(tree.children[2], inputs, grad_inputs)
  end
end


function BinaryTreeLSTM:parameters()
  return self.modules[1]:parameters()
end


----------------------------------------------------
-- helper functions
----------------------------------------------------

function BinaryTreeLSTM:acc_grad_output(tree, x)
  if #x == 1 then
    if tree[self.grad_output_name] == nil then
      tree[self.grad_output_name] = {x[1]:clone():cuda(), self.empty_output:clone():cuda()}
    else
      tree[self.grad_output_name][1]:add(x[1])
    end
  elseif #x == 2 then
    if tree[self.grad_output_name] == nil then
      tree[self.grad_output_name] = {x[1]:clone():cuda(), x[2]:clone():cuda()}
    else
      tree[self.grad_output_name][1]:add(x[1])
      tree[self.grad_output_name][2]:add(x[2])
    end
  else
    assert(#x==1 or #x==2, "wrong number of tensors for accumulating grad output")
  end
  return tree[self.grad_output_name]
end


function BinaryTreeLSTM:get_children_outputs(tree)
  local lh, lc, rh, rc
  lh = tree.children[1][self.output_name][1]
  lc = tree.children[1][self.output_name][2]
  rh = tree.children[2][self.output_name][1]
  rc = tree.children[2][self.output_name][2]
  return lh, lc, rh, rc
end
