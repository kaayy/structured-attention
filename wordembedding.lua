--[[

  Loads word embeddings from text word2vec format. The loaded word embeddings are cached.

--]]

require("torch")
require("pl")

require("utils")

torch.class("WordEmbedding")

function WordEmbedding:__init(path)
  self.max_word_width = 1024
  self.OOV_SYM = "<OOV>"

  local cache_path = path .. ".t7"
  if not paths.filep(cache_path) then
    printerr("Loading embedding from raw file...")
    self.vocab, self.embeddings = self:load_from_raw(path, cache_path)
  else
    printerr("Loading embedding from cache file...")
    local cache = torch.load(cache_path)
    self.vocab, self.embeddings = cache[1], cache[2]
  end

  self.word2idx = nil

  printerr(#self.vocab .. " words loaded.")

end


function WordEmbedding:get_word_idx(word)
  if self.word2idx == nil then
    self.word2idx = {}
    for i, v in ipairs(self.vocab) do
      self.word2idx[v] = i
    end
  end
  return self.word2idx[word]
end


function WordEmbedding:load_from_raw(path, cache_path)
  function read_string(file)
    -- helper function that reads a word
    local str = {}
    for i =  1, self.max_word_width do
      local char = file:readChar()
      if char == 32 or char == 10 or char == 0 then
        break
      else
        str[#str+1] = char
      end
    end
    str = torch.CharStorage(str)
    return str:string()
  end
  local file = torch.DiskFile(path, "r")
  file:ascii()
  local num_words = file:readInt()
  local num_dim = file:readInt()

  local vocab = {}
  local embeddings = torch.Tensor(num_words, num_dim)

  for i = 1, num_words do
    local word = read_string(file)
    local vecstorage = file:readFloat(num_dim)
    local vec = torch.FloatTensor(num_dim)
    vec:storage():copy(vecstorage)
    vocab[i] = word
    embeddings[{{i}, {}}] = vec
  end

  printerr("Writing to embedding to cache...")
  torch.save(cache_path, {vocab, embeddings})

  return vocab, embeddings
end


function WordEmbedding:save(path)
  local num_words = #self.vocab
  local dim = self.embeddings:size(2)
  local f = io.open(path, "w")
  f:write(string.format("%d %d\n", num_words, dim))
  for i=1, num_words do
    local w = self.vocab[i]
    local vec = stringx.join(" ", COMP 'tostring(x) for x' (self.embeddings[i]:float():totable()))
    f:write(string.format("%s %s\n", w, vec))
  end
  f:close()
end


function WordEmbedding:trim_by_counts(word_counts)
  -- removes words w/o counts
  local trimmed_vocab = {}
  trimmed_vocab[#trimmed_vocab + 1] = self.OOV_SYM

  for i, w in ipairs(self.vocab) do
    if word_counts[w] ~= nil then
      trimmed_vocab[#trimmed_vocab + 1] = w
    end
  end

  local trimmed_embeddings = torch.Tensor(#trimmed_vocab, self.embeddings:size(2))

  for i, w in ipairs(trimmed_vocab) do
    if w == self.OOV_SYM then
      trimmed_embeddings[i] = (torch.rand(self.embeddings:size(2)) - 0.5) / 10
    else
      trimmed_embeddings[i] = self.embeddings[self:get_word_idx(w)]
    end
  end

  self.vocab = trimmed_vocab
  self.embeddings = trimmed_embeddings
  self.word2idx = nil
end


function WordEmbedding:extend_by_counts(word_counts)
  -- adds words in the counts
  local extended_vocab = {}
  for i, w in ipairs(self.vocab) do extended_vocab[#extended_vocab + 1] = w end

  local dict = Set(self.vocab)
  for w, c in pairs(word_counts) do
    if not dict[w] then extended_vocab[#extended_vocab + 1] = w end
  end

  local extended_embeddings = torch.Tensor(#extended_vocab, self.embeddings:size(2))
  for i, w in ipairs(extended_vocab) do
    if w == self.OOV_SYM then
      extended_embeddings[i] = (torch.rand(self.embeddings:size(2)) - 0.5) / 10
    elseif self:get_word_idx(w) ~= nil then
      extended_embeddings[i] = self.embeddings[self:get_word_idx(w)]
    else
      extended_embeddings[i] = (torch.rand(self.embeddings:size(2)) - 0.5) / 10
    end
  end

  self.vocab = extended_vocab
  self.embeddings = extended_embeddings
  self.word2idx = nil
end


function WordEmbedding:convert(words)
  -- converts the words to a vector of indices of word embeddings
  indices = torch.IntTensor(#words)
  for i, w in pairs(words) do
    idx = self:get_word_idx(w)
    if idx == nil then idx = self:get_word_idx(self.OOV_SYM) end
    indices[i] = idx
  end
  return indices
end
