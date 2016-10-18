--[[

  Tree structure.

--]]

require("torch")
require("pl")
require("math")
local moses = require("moses")


torch.class("Tree")

function Tree:__init(val, children)
  self.val = val
  self.children = children
end

function Tree:__tostring()
  if self.val ~= nil then
    return self.val
  else
    return "( " .. stringx.join(" ",
                                moses.map(self.children,
                                          function (k, v) return tostring(v) end
                               )) .. " )"
  end
end

function Tree:parse(treestr, prune_last_period)
  --[[ Loads a tree from the input string.
    Args:
      treestr: tree string in parentheses form.
    Returns:
      An instance of Tree.
  --]]
  local _, t = Tree:_parse(treestr .. " ", 1)
  if prune_last_period and false then
    t:prune_last_period()
  end
  t:mark_leaf_id()
  t:mark_postorder()
  return t
end

function Tree:_parse(treestr, index)
  assert(stringx.at(treestr, index) == "(", "Invalid tree string " .. treestr .. " at " .. index)
  index = index + 1
  local children = {}
  while stringx.at(treestr, index) ~= ")" do
    if stringx.at(treestr, index) == "(" then
      index, t = Tree:_parse(treestr, index)
      children[#children + 1] = t
    else
      -- leaf
      local rpos = math.min(stringx.lfind(treestr, " ", index), stringx.lfind(treestr, ")", index))
      local leaf_word = treestr:sub(index, rpos-1)
      if leaf_word ~= "" then
        children[#children + 1] = Tree(leaf_word, {})
      end
      index = rpos
    end

    if stringx.at(treestr, index) == " " then
      index = index + 1
    end

  end

  assert(stringx.at(treestr, index) == ")", "Invalid tree string " .. treestr .. " at " .. index)

  local t = Tree(nil, children)
  return index+1, t
end


function Tree:mark_leaf_id()
  -- converts the tree leafs from words to indices in the sentence
  local count = 1
  self:inorder_traverse(
    function (subtree)
      if subtree.val ~= nil then
        subtree.leaf_id = count
        count = count + 1
      end
    end
  )
end


function Tree:mark_postorder()
  local count = 1
  self:postorder_traverse(
    function (subtree)
      subtree.postorder_id = count
      count = count + 1
    end
  )
end


function Tree:get_sentence(accumulated)
  -- get words from leafs to form the sentence
  local sent = accumulated or {}
  if self.val ~= nil then  -- leaf
    sent[#sent + 1] = self.val
    return sent
  else
    for i, v in ipairs(self.children) do
      sent = v:get_sentence(sent)
    end
    return sent
  end
end


function Tree:postorder_traverse(func)
  for i, v in ipairs(self.children) do v:postorder_traverse(func) end
  func(self)
end


function Tree:preorder_traverse(func)
  func(self)
  for i, v in ipairs(self.children) do v:preorder_traverse(func) end
end


function Tree:inorder_traverse(func)
  if self.val ~= nil then
    func(self)
  else
    assert(#self.children == 2,  "wrong number of children")
    self.children[1]:inorder_traverse(func)
    func(self)
    self.children[2]:inorder_traverse(func)
  end
end


function Tree:prune_last_period()
  if self.val == nil then
    if self.children[2].val == "." then
      self.val = self.children[1].val
      self.children = self.children[1].children
    else
      self.children[2]:prune_last_period()
    end
  end
end


function Tree:prune(test_func)
  -- return true is this tree node needs to be pruned
  if self.val == nil then
    -- internal node
    local leftprune = self.children[1]:prune(test_func)
    local rightprune = self.children[2]:prune(test_func)
    if leftprune == nil and rightprune == nil then
      -- both left and right are pruned
      return nil
    elseif leftprune == nil then return rightprune
    elseif rightprune == nil  then return leftprune
    else
      self.children[1] = leftprune
      self.children[2] = rightprune
      return self
    end
  elseif test_func(self.val) then
    -- leaf node
    return nil
  else
    return self
  end
end
