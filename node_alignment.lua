--[[

  Calculates node repr with attention

--]]


require("torch")
require("cutorch")
require("nn")
require("cunn")
require("nngraph")

require("utils")
require("ReplicateAs")

torch.class("NodeAlignment")


function NodeAlignment:__init(config)
  self.input_dim = config.input_dim
  self.output_dim = config.output_dim
  self.treelstm = config.treelstm
  self.nullalignment = config.nullalignment
  self.new_attention_module = NodeAlignment.allocate_attention_module
  self.new_similarity_module = NodeAlignment.allocate_similarity_module
  if self.nullalignment then
    -- a fake null node pass through the tree lstm
    self.null_lstm_repr = nil
    self.null_module = config.treelstm:new_module():cuda()
    share_params(self.null_module, config.treelstm.modules[1])
    self.dropout = nn.Dropout(0.2):cuda()
    self.null_repr = torch.zeros(300):uniform(-0.05, 0.05):cuda()
    self.null_repr_grad = torch.zeros(300):cuda()
    self.empty_children = config.treelstm.empty_output
  end
  if config.extend then
    self.new_attention_module = NodeAlignment.allocate_extended_attention_module
    self.new_similarity_module = NodeAlignment.allocate_extended_similarity_module
  end
  self.attention_modules = {self.new_attention_module(self.input_dim)}
  self.similarity_modules = {self.new_similarity_module(self.input_dim, self.output_dim)}

  self.norm_module = nn.Normalize(1):cuda()
end


function NodeAlignment:get_modules(module_id)
  if #self.attention_modules < module_id then
    local new_att_module = self.new_attention_module(self.input_dim)
    local new_sim_module = self.new_similarity_module(self.input_dim, self.output_dim)
    share_params(new_att_module, self.attention_modules[1])
    share_params(new_sim_module, self.similarity_modules[1])
    self.attention_modules[#self.attention_modules + 1] = new_att_module
    self.similarity_modules[#self.similarity_modules + 1] = new_sim_module
    return self:get_modules(module_id)
  else
    return self.attention_modules[module_id], self.similarity_modules[module_id]
  end
end


function NodeAlignment.allocate_attention_module(input_dim)
  local Y = nn.Identity()()
  local h = nn.Identity()()

  local repH = nn.ReplicateAs(){h, Y}
  local M = nn.Tanh()(
    nn.Linear(input_dim, input_dim)(
      nn.Abs()(nn.CSubTable(){
                 Y, repH})))

  local a = nn.SoftMax()(nn.View()(nn.LinearNoBias(input_dim, 1)(M)))

  return nn.gModule({Y, h}, {a}):cuda()
end


function NodeAlignment.allocate_extended_attention_module(input_dim)
  local Y = nn.Identity()()
  local h = nn.Identity()()

  local repH = nn.ReplicateAs(){h, Y}
  local M = nn.Tanh()(
    nn.CAddTable(){
      nn.Linear(input_dim, input_dim)(
        nn.Abs()(nn.CSubTable(){Y, repH})),
      nn.LinearNoBias(input_dim, input_dim)(Y),
      nn.LinearNoBias(input_dim, input_dim)(repH)
                     })

  local a = nn.SoftMax()(nn.View()(nn.LinearNoBias(input_dim, 1)(M)))

  return nn.gModule({Y, h}, {a}):cuda()

end


function NodeAlignment.allocate_similarity_module(input_dim, output_dim)
  local Y = nn.Identity()()
  local a = nn.Identity()()
  local h = nn.Identity()()

  local hsrc = nn.View()(nn.MM(){nn.Transpose({1, 2})(Y), nn.Reshape(1)(a)})
  local r = nn.CAddTable(){nn.Linear(input_dim, output_dim)(hsrc),
                           nn.LinearNoBias(input_dim, output_dim)(h),}

  return nn.gModule({Y, a, h}, {r}):cuda()
end


function NodeAlignment.allocate_extended_similarity_module(input_dim, output_dim)
  local Y = nn.Identity()()
  local a = nn.Identity()()
  local h = nn.Identity()()

  local hsrc = nn.View()(nn.MM(){nn.Transpose({1, 2})(Y), nn.Reshape(1)(a)})
  local r = nn.ReLU()(nn.CAddTable(){
    nn.Linear(input_dim, output_dim)(nn.Abs()(nn.CSubTable(){hsrc, h})),
    nn.LinearNoBias(input_dim, output_dim)(hsrc),
    nn.LinearNoBias(input_dim, output_dim)(h),})

  return nn.gModule({Y, a, h}, {r}):cuda()
end


function NodeAlignment:forward(ltree, rtree)
  self.Y = self:aggregate_MR(ltree)
  local softatt = torch.zeros(rtree.postorder_id, self.Y:size(1))
  rtree:postorder_traverse(
    function (subtree)
      local att, _ = self:get_modules(subtree.postorder_id)
      local rep = subtree.lstm_output[1]
      local a = att:forward{self.Y, rep}
      subtree.attention = a
      softatt[subtree.postorder_id]:copy(a)
    end
  )

  softatt = softatt:cuda()

  rtree:postorder_traverse(
    function (subtree)

      local a = subtree.attention
      local rep = subtree.lstm_output[1]

      local _, sim = self:get_modules(subtree.postorder_id)

      local mr = sim:forward{self.Y, a, rep}
      subtree.alignment_output = mr
    end
  )
end


function NodeAlignment:backward(ltree, rtree, loss)
  local Y_grad = torch.zeros(self.Y:size()):cuda()
  rtree:postorder_traverse(
    function (subtree)
      local att, sim = self:get_modules(subtree.postorder_id)
      local rep = subtree.lstm_output[1]
      local sim_grad = sim:backward({self.Y, subtree.attention, rep}, subtree.alignment_grad_output)

      subtree.sim_grad = sim_grad
      Y_grad:add(sim_grad[1])
      self.treelstm:acc_grad_output(subtree, {sim_grad[3]})
    end
  )

  rtree:postorder_traverse(
    function (subtree)
      local att, sim = self:get_modules(subtree.postorder_id)
      local rep = subtree.lstm_output[1]
      local sim_grad = subtree.sim_grad

      local att_grad
      att_grad = att:backward({self.Y, rep}, sim_grad[2])

      Y_grad:add(att_grad[1])
      self.treelstm:acc_grad_output(subtree, {att_grad[2]})
    end
  )
  self:acc_MR_grad(ltree, Y_grad)

end


function NodeAlignment:aggregate_MR(tree)
  -- aggregate the meaning representation vectors in each tree node as a matrx
  local num_nodes = tree.postorder_id
  if self.nullalignment then
    num_nodes = num_nodes + 1
  end
  local Ytab = torch.zeros(num_nodes, self.input_dim)

  tree:postorder_traverse(
    function (subtree)
      Ytab[subtree.postorder_id]:copy(subtree.lstm_output[1])
    end
  )

  if self.nullalignment then
    self.dropout_null = self.dropout:forward(self.null_repr)
    self.null_lstm_repr = self.null_module:forward{self.dropout_null,
                                                   self.empty_children, self.empty_children,
                                                   self.empty_children, self.empty_children}
    Ytab[num_nodes]:copy(self.null_lstm_repr[1])
  end

  return torch.Tensor(Ytab):cuda()
end


function NodeAlignment:acc_MR_grad(tree, Y_grad)
  tree:postorder_traverse(
    function (subtree)
      self.treelstm:acc_grad_output(subtree, {Y_grad[subtree.postorder_id]})
    end
  )
  if self.nullalignment then
    local null_grad_input = self.null_module:backward({self.dropout_null,
                                                       self.empty_children, self.empty_children,
                                                       self.empty_children, self.empty_children},
      {Y_grad[Y_grad:size(1)], self.empty_children})
    local dropout_grad = self.dropout:backward(self.null_repr, null_grad_input[1])
    self.null_repr_grad:add(dropout_grad)
  end
end


function NodeAlignment:acc_grad_output(tree, grad_output)
  tree.alignment_grad_output = grad_output
end


function NodeAlignment:training()
  self.train = true
end

function NodeAlignment:evaluate()
  self.train = false
end

function NodeAlignment:parameters()
  local params, grad_params = {}, {}
  local ap, ag = self.attention_modules[1]:parameters()
  tablex.insertvalues(params, ap)
  tablex.insertvalues(grad_params, ag)
  local sp, sg = self.similarity_modules[1]:parameters()
  tablex.insertvalues(params, sp)
  tablex.insertvalues(grad_params, sg)
  if self.nullalignment then
    params[#params+1] = self.null_repr
    grad_params[#grad_params+1] = self.null_repr_grad
  end
  return params, grad_params
end
