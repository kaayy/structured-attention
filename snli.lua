--[[
  
  Loads SNLI entailment dataset.

--]]

require("torch")
require("pl")
local moses = require("moses")

require("utils")
require("tree")

torch.class("SNLI")

function SNLI:__init(snli_path_prefix, train_size, lower_case, verbose)
  self.num_relations = 3
  self.relations = {["contradiction"] = 1, ["neutral"] = 2, ["entailment"] = 3}
  self.rev_relations = {}
  for r, i in pairs(self.relations) do self.rev_relations[i] = r end
  self.train_size = train_size
  self.lower_case = lower_case
  self.verbose = verbose

  self.train_word_counts = {}
  self.word_counts = {}

  if snli_path_prefix ~= nil then
    self.verbose = false
    self.train = self:_load_data_file(snli_path_prefix .. "train.txt", self.train_word_counts)
    for k, v in pairs(self.train_word_counts) do self.word_counts[k] = v end
    self.dev = self:_load_data_file(snli_path_prefix .. "dev.txt", self.word_counts)

    self.verbose = verbose

    if self.train_size > 0 then
      self.train = tablex.sub(self.train, 1, self.train_size)
    end

    if self.verbose then
      printerr(string.format("SNLI train: %d pairs", #self.train))
      printerr(string.format("SNLI dev: %d pairs", #self.dev))
    end
  end

end


function SNLI:inc_word_counts(word, counter)
  if counter[word] ~= nil then
    counter[word] = counter[word] + 1
  else
    counter[word] = 1
  end
end


function SNLI:_load_data_file(file_path, word_counter)
  local data = {}
  for i, line in seq.enum(io.lines(file_path)) do
    local line_split = stringx.split(line, "\t")
    local gold_label = line_split[1]
    if self.relations[gold_label] ~= nil then
      if not pcall(
        function ()
          local premise = stringx.split(line_split[2])
          local hypothese = stringx.split(line_split[3])
          if self.lower_case then
            premise = moses.map(premise, function(i, v) return string.lower(v) end)
            hypothese = moses.map(hypothese, function(i,v) return string.lower(v) end)
          end

          for i, v in ipairs(premise) do self:inc_word_counts(v, word_counter) end
          for i, v in ipairs(hypothese) do self:inc_word_counts(v, word_counter) end

          local ptree_str = stringx.join(" ", premise)
          local htree_str = stringx.join(" ", hypothese)
          local ptree = Tree:parse(ptree_str)
          local htree = Tree:parse(htree_str)
          data[#data+1] = {["label"] = self.relations[gold_label],
            ["id"] = #data+1,
            ["premise"] = ptree_str, ["hypothese"] = htree_str}
        end
      ) then
        if self.verbose then
          printerr("error loading " .. line)
        end
      end
    end
  end
  return data
end
