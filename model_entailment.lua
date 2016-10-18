--[[

  Structured Attention Model.

--]]

require("torch")
require("cutorch")
require("nn")
require("cunn")
require("nngraph")
require("optim")
require("pl")

require("fastbinarytreelstm")
require("simpleprofiler")
require("utils")
require("node_alignment")

torch.class("StructuredEntailmentModel")


function StructuredEntailmentModel:__init(config)
  self.word_emb = config.word_emb
  self.word_dim = self.word_emb.embeddings:size(2)
  self.repr_dim = config.repr_dim
  self.num_relations = config.num_relations
  self.learning_rate = config.learning_rate
  self.batch_size = config.batch_size
  self.dropout = config.dropout
  self.interactive = config.interactive
  self.words_from_embbedding = config.words_from_embbedding
  self.update_oov_only = config.update_oov_only
  self.hiddenrel = config.hiddenrel
  self.verbose = config.verbose
  self.dataset = config.dataset

  if self.verbose then
    printerr("------------------------------")
    printerr("Model parameters:")
    printerr("repr dim " .. self.repr_dim)
    printerr("hidden rel " .. self.hiddenrel)
    printerr("learning rate " .. self.learning_rate)
    printerr("dropout " .. self.dropout)
    printerr("batch size " .. self.batch_size)
    printerr("interactive " .. tostring(self.interactive))
    printerr("OOV only " .. tostring(self.update_oov_only))
  end

  self.relation_module = self:new_relation_mapping_module():cuda()

  self.optim_state = { learningRate = self.learning_rate }

  -- layers

  self.emb_p = nn.LookupTable(#self.word_emb.vocab, self.word_dim):cuda()
  self.emb_h = nn.LookupTable(#self.word_emb.vocab, self.word_dim):cuda()

  self.dropout_p = nn.Dropout(self.dropout):cuda()
  self.dropout_h = nn.Dropout(self.dropout):cuda()

  self.treelstm = BinaryTreeLSTM{
    name = "lstm",
    input_dim = self.word_dim,
    output_dim = self.repr_dim,
    get_input = function (inputs, tree)
      if tree.leaf_id then
        return inputs[tree.leaf_id]
      else
        return nil
      end
    end,
    acc_grad_input = function (grad_inputs, tree, grad_input)
      if tree.leaf_id ~= nil then
        grad_inputs[tree.leaf_id]:add(grad_input)
      end
    end
  }

  self.alignment = NodeAlignment{
    input_dim = self.repr_dim,
    output_dim = self.repr_dim,
    treelstm = self.treelstm,
    nullalignment = true,
    extend = true
  }

  self.entailment = BinaryTreeLSTM{
    input_dim = self.repr_dim,
    output_dim = self.hiddenrel,
    name = "entailment",
    get_input = function (_, tree) return tree.alignment_output end,
    acc_grad_input = function (_, tree, grad_input) self.alignment:acc_grad_output(tree, grad_input) end
  }

  self.criterion = nn.ClassNLLCriterion():cuda()


  local modules = nn.Parallel()
    :add(self.emb_p)
    :add(self.treelstm)
    :add(self.alignment)
    :add(self.entailment)
    :add(self.relation_module)

  self.params, self.grad_params = modules:getParameters()
  self.params:uniform(-0.05, 0.05)
  print(getTensorSize(self.params))

  self.b = 0

  self.emb_p.weight:copy(self.word_emb.embeddings):cuda()
  share_params(self.emb_h, self.emb_p)

  self.modules = {self.emb_p, self.emb_h,
                  self.dropout_p, self.dropout_h,
                  self.treelstm,
                  self.alignment, self.entailment,
                  self.relation_module}

end


function StructuredEntailmentModel:new_relation_mapping_module()
  local e = nn.Identity()()
  local ret = nn.LogSoftMax()(nn.Linear(self.hiddenrel, self.num_relations)(e))
  return nn.gModule({e}, {ret})
end


function StructuredEntailmentModel:set_training(train)
  self.is_training = train
  for i, m in ipairs(self.modules) do
    if train then
      m:training()
    else
      m:evaluate()
    end
  end
end


function StructuredEntailmentModel:annotate(tree, reftree)
  reftree:postorder_traverse(
    function (subtree)
      print(subtree.postorder_id, subtree)
    end
  )
  -- annotate a processed hypothesis tree
  tree:postorder_traverse(
    function (subtree)
      local label = self.relation_module:forward(subtree.entailment_output[1])
      local tab = {}
      for i=1,label:size(1) do tab[i] = tostring(torch.exp(label[i])) end
      local values, indices = torch.sort(label)
      print(string.format("**** node %d %s : %d(%s) ****",
                          subtree.postorder_id, tostring(subtree),
                          indices[3], self.dataset.rev_relations[indices[3]]))
      print("\tentailment:", stringx.join(" ", tab))
      if self.show_alignment or true then
        tab = {}
        for i=1,subtree.attention:size(1) do
          tab[i] = string.format("%d:%.4f", i, subtree.attention[i])
        end
        print("\talignment:", stringx.join(" ", tab))
      end
    end
  )
end


function StructuredEntailmentModel:train(examples)
  self:set_training(true)
  local num_examples = #examples
  local zeros = torch.zeros(self.repr_dim):cuda()

  local total_loss = 0

  local correct = 0

  local report_freq = num_examples / 100
  local report_point = 0

  for i = 1, num_examples, self.batch_size do
    if self.interactive then
      xlua.progress(i, num_examples)
    else
      if i > report_point then
        printerr(i .. " ", "")
        report_point =  report_point + report_freq
      end
    end

    local batch_size = math.min(i + self.batch_size - 1, num_examples) - i + 1

    local train_batch = function(x)
      self.grad_params:zero()
      self.emb_p:zeroGradParameters()
      local loss = 0
      for j = 1, batch_size do
        local idx = i + j - 1

        -- load tree from tree string, get sentence from tree, and convert original tree leaf words to indices
        local example = examples[idx]

        local info = self:process_one_example(example)
        loss = loss + info.loss
        if info.correct then correct = correct + 1 end

      end
      loss = loss / batch_size
      total_loss = total_loss + loss

      self.b = self.b * 0.9 + loss * 0.1

      if self.update_oov_only then
        local _, emb_grad = self.emb_p:parameters()
        emb_grad[1]:narrow(1,1,self.words_from_embbedding):zero()
      end
      self.grad_params:div(batch_size)

      cutorch.synchronize()

      return loss, self.grad_params
    end

    optim.adam(train_batch, self.params, self.optim_state)

  end

  printerr(string.format("\nAt training acc %f total loss %f params norm %f",
                         correct / num_examples, total_loss, self.params:norm()))

  local info = {
    ["acc"] = correct/num_examples,
    ["loss"] = total_loss}

  return info
end


function StructuredEntailmentModel:process_one_example(example)
  local ret = {}
  local reference = example["label"]
  local ltreestr, rtreestr = example["premise"], example["hypothese"]
  local ltree, rtree = Tree:parse(ltreestr), Tree:parse(rtreestr)
  local lsent = self.word_emb:convert(ltree:get_sentence())
  local rsent = self.word_emb:convert(rtree:get_sentence())

  ret.premise = ltree
  ret.hypothesis = rtree

  local verbose = false

  local ltree_offset = lsent:size(1)

  local linputs0 = self.emb_p:forward(lsent)
  local rinputs0 = self.emb_h:forward(rsent)
  local linputs = self.dropout_p:forward(linputs0)
  local rinputs = self.dropout_h:forward(rinputs0)

  -- get sentence representations
  local lrep = self.treelstm:forward(ltree, linputs)
  local rrep = self.treelstm:forward(rtree, rinputs, ltree_offset)

  if verbose then print("repr", ltree.lstm_output[1]:norm(), rtree.lstm_output[1]:norm()) end

  -- compute relatedness
  self.alignment:forward(ltree, rtree)
  local entailment_repr = self.entailment:forward(rtree)

  local output = self.relation_module:forward(entailment_repr)

  local values, indices = torch.sort(output)
  local correct = reference == indices[3]

  ret.correct = correct
  ret.predicted = indices[3]

  if self.is_training then
    -- compute loss and backpropagate
    local example_loss = self.criterion:forward(output, reference)
    ret.loss = example_loss

    local sim_grad = self.criterion:backward(output, reference)
    local rep_grad = self.relation_module:backward(entailment_repr, sim_grad)
    if verbose then print("repr grad", rep_grad:norm()) end

    self.entailment:acc_grad_output(rtree, {rep_grad})
    self.entailment:backward(rtree)
    if verbose then print("entailment grad", rtree.alignment_grad_output:norm()) end

    self.alignment:backward(ltree, rtree, example_loss - self.b)

    local linput_grads = torch.zeros(linputs:size()):cuda()
    self.treelstm:backward(ltree, linputs, linput_grads)
    local rinput_grads = torch.zeros(rinputs:size()):cuda()
    self.treelstm:backward(rtree, rinputs, rinput_grads)

    local linput_grads0 = self.dropout_p:backward(linputs0, linput_grads)
    local rinput_grads0 = self.dropout_h:backward(rinputs0, rinput_grads)
    self.emb_p:backward(lsent, linput_grads0)
    self.emb_h:backward(rsent, rinput_grads0)
  end

  return ret
end



function StructuredEntailmentModel:checkParams()
  print("params for modules")
  local embp, _ = self.emb_p:parameters()
  print("emb p", getTensorTableNorm(embp))
  local embh, _ = self.emb_h:parameters()
  print("emb h", getTensorTableNorm(embh))
  local treelstm, _ = self.treelstm.modules[1]:parameters()
  print("treelstm p")
  for i, v in ipairs(treelstm) do
    print(i, getTensorSize(v), v:norm())
  end
  local alignment, _ = self.alignment:parameters()
  print("alignment")
  for i, v in ipairs(alignment) do
    print(i, getTensorSize(v), v:norm())
    if tensorSize(v) ==  1 then print(v) end
  end
  local entailment, _ = self.entailment:parameters()
  print("entailment")
  for i, v in ipairs(entailment) do
    print(i, getTensorSize(v), v:norm())
    if tensorSize(v) ==  1 then print(v) end
  end
  local rel, _ = self.relation_module:parameters()
  print("relation", getTensorTableNorm(rel))
end


function StructuredEntailmentModel:evaluate(examples, verbose)
  self:set_training(false)
  local correct = 0
  local num_examples = #examples
  local report_freq = num_examples / 100
  local report_point = 0

  for i = 1, num_examples do
    if self.interactive then
      xlua.progress(i, num_examples)
    else
      if i > report_point then
        printerr(i .. " ", "")
        report_point = report_point + report_freq
      end
    end

    local example = examples[i]
    local reference = example.label
    local info = self:process_one_example(example)

    if info.correct then correct = correct + 1
    elseif verbose then
      print(string.format("error %d\t%s->%s\t%s\t%s", i,
                          self.dataset.rev_relations[info.predicted],
                          self.dataset.rev_relations[reference],
                          example.premise, example.hypothese))
    end

    if verbose and false then
      -- print status of the hypothesis tree
      self:annotate(info.hypothesis, info.premise)
    end
  end

  printerr("")

  local info = {["acc"] = correct / num_examples}

  return info
end


function StructuredEntailmentModel:aggregateMR(tree)
  -- aggregate the meaning representation vectors in each tree node as a matrx
  local num_nodes = tree.postorder_id
  local Ytab = torch.zeros(num_nodes, self.repr_dim)
  tree:postorder_traverse(
    function (subtree)
      Ytab[subtree.postorder_id]:copy(subtree.lstm_output[1])
    end
  )

  return torch.Tensor(Ytab):cuda()
end

function StructuredEntailmentModel:accMR(tree, Y_grad)
  assert(Y_grad:size(1) == tree.postorder_id, "Sizes of Y grad and tree nodes do not match")
  tree:postorder_traverse(
    function (subtree)
      self.treelstm:acc_grad_output(subtree,
                                      {Y_grad[subtree.postorder_id]}) end
  )
end
