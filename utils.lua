--[[
  
  Utility functions.

--]]

require("torch")
require("cutorch")
require("nn")
require("cunn")
require("nngraph")
require("pl")

-- list comprehension operator
COMP = require("pl.comprehension").new()


function printerr(msg, newline)
  local suffix = newline or "\n"
  io.stderr:write(tostring(msg) .. suffix):flush()
end


function getTensorSize(tensor, separator)
  local sep = separator or " "
  local ret = {}
  for i = 1, tensor:dim() do
    ret[i] = tensor:size(i)
  end
  return stringx.join(sep, ret)
end


-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
                               'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end


function getTensorDataAddress(x)
  return string.format("%x+%d", torch.pointer(x:storage():data()), x:storageOffset())
end


function getTensorTableNorm(t)
  local ret = 0
  for i, v in ipairs(t) do
    ret = ret + v:norm()^2
  end
  return math.sqrt(ret)
end


function incCounts(counter, key)
  if counter[key] ~= nil then
    counter[key] = counter[key] + 1
  else
    counter[key] = 1
  end
end


function tableLength(tab)
  local count = 0
  for _ in pairs(tab) do count = count + 1 end
  return count
end


function repeatTensorAsTable(tensor, count)
  local ret = {}
  for i = 1, count do ret[i] = tensor end
  return ret
end


function flattenTable(tab)
  local ret = {}
  for _, t in ipairs(tab) do
    if torch.type(t) == "table" then
      for _, s in ipairs(flattenTable(t)) do
        ret[#ret + 1] = s
      end
    else
      ret[#ret + 1] = t
    end
  end
  return ret
end


function getTensorTableSize(tab, separator)
  local sep = separator or " "
  local ret = {}
  for i, t in ipairs(tab) do
    ret[i] = getTensorSize(t, "x")
  end
  return stringx.join(sep, ret)
end


function vectorStringCompact(vec, separator)
  local sep = separator or " "
  local ret = {}
  for i = 1, vec:size(1) do
    ret[i] = string.format("%d:%.4f", i, vec[i])
  end
  return stringx.join(sep, ret)
end


function tensorSize(tensor)
  local size = 1
  for i=1, tensor:dim() do size = size * tensor:size(i) end
  return size
end

-- http://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html
StopWords = Set{"a", "an", "and", "are", "as", "at", "be", "by",
                "for", "from", "has", "in", "is", "of", "on", "that",
                "the", "to", "was", "were", "will", "with", "."}

function isStopWord(word)
  return StopWords[word]
end
