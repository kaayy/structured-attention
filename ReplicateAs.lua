--[[
  Replicate the first tensor with the shape of the second tensor.
  Usage:
   local rep = nn.ReplicateAs()
   rep:forward(tensor_to_be_replicated, tensor_providing_shape)
--]]

require("nn")

local ReplicateAs, parent = torch.class('nn.ReplicateAs','nn.Module')

function ReplicateAs:__init(dim, ndim)
   parent.__init(self)
   self.dim = dim or 1
   self.ndim = ndim
   assert(self.dim > 0, "Can only replicate across positive integer dimensions.")
end

function ReplicateAs:updateOutput(input)
  self.dim = self.dim or 1 --backwards compatible
   assert(
     self.dim <= input[1]:dim()+1,
      "Not enough input dimensions to replicate along dimension " ..
      tostring(self.dim) .. ".")
   local batchOffset = self.ndim and input[1]:dim() > self.ndim and 1 or 0
   local rdim = self.dim + batchOffset
   local sz = torch.LongStorage(input[1]:dim()+1)
   sz[rdim] = input[2]:size()[1]
   for i = 1,input[1]:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      sz[i+offset] = input[1]:size(i)
   end
   local st = torch.LongStorage(input[1]:dim()+1)
   st[rdim] = 0
   for i = 1,input[1]:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      st[i+offset] = input[1]:stride(i)
   end
   self.output = input[1].new(input[1]:storage(),input[1]:storageOffset(),sz,st)
   return self.output
end

function ReplicateAs:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input[1]):zero()
  local batchOffset = self.ndim and input[1]:dim() > self.ndim and 1 or 0
  local rdim = self.dim + batchOffset
  local sz = torch.LongStorage(input[1]:dim()+1)
  sz[rdim] = 1
  for i = 1,input[1]:dim() do
    local offset = 0
    if i >= rdim then
      offset = 1
    end
    sz[i+offset] = input[1]:size(i)
  end
  local gradInput = self.gradInput:view(sz)
  gradInput:sum(gradOutput, rdim)
  local gradInputShape = torch.zeros(input[2]:size()):cuda()
  return {self.gradInput, gradInputShape}
end

--[[
require("torch")
require("nn")
s = torch.Tensor{1,2}
p = torch.Tensor{{1, 2}, {3, 4}}
rep = nn.ReplicateAs()
print(rep:forward{s, p})

s = torch.Tensor{{1, 2}}
q = torch.Tensor{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}
r = rep:forward{s, q}
print(r)
mm = nn.MM()
r2 = mm:forward{r, q}
print(r2)
mm2 = nn.MM(false, true)
r3 = mm2:forward{r2, r}
print(r3)
--]]
