--[[

  The trainer.

--]]

require("torch")
require("cutorch")
require("pl")

require("utils")
require("wordembedding")
require("snli")
require("simpleprofiler")
require("model_entailment")

torch.manualSeed(123)

cutorch.manualSeedAll(123)

local args = lapp [[
Training script for sentence entailment on SNLI.
  -t,--train_size (default 0)    # of samples used in training
  --dim    (default 150)        LSTM memory dimension
  -e,--epochs (default 30)         Number of training epochs
  -r,--learning_rate (default 0.001) Learning rate
  -m,--batch_size (default 32)  Batch size
  --hiddenrel (default 150) # of hidden relations
  --dataset_prefix (default sampledata/)  Prefix of path to dataset
  -d,--dropout (default 0.2)  Dropout rate
  -w,--word_embedding (default sampledata/wordembedding)    Path to word embedding
  --gpu (default 1)  The gpu device to use
  --interactive (default true) Show progress interactively
  --dump (default nil) Weights dump
  --eval (default nil) Evaluate weights
  --oovonly (default true) Update OOV embeddings only
]]

cutorch.setDevice(args.gpu)

torch.class("Trainer")

function Trainer:__init(verbose)
  self.verbose = verbose or true
  if self.verbose then printerr("Word embedding path " .. args.word_embedding) end
  self.word_embedding = WordEmbedding(args.word_embedding)

  if self.verbose then printerr("Dataset prefix " .. args.dataset_prefix) end

  self.dump = args.dump

  self.data = SNLI(args.dataset_prefix,
                   args.train_size,  -- train_size
                   true, -- lower_case
                   true)  -- verbose

  -- trim the word embeddings to contain only words in the dataset
  if self.verbose then
    printerr("Before trim word embedding, " .. self.word_embedding.embeddings:size(1) .. " words")
  end
  self.word_embedding:trim_by_counts(self.data.word_counts)
  local words_from_embedding = self.word_embedding.embeddings:size(1)
  if self.verbose then
    printerr("After trim word embedding, " .. words_from_embedding .. " words")
  end

  self.word_embedding:extend_by_counts(self.data.train_word_counts)

  if self.verbose then
    printerr("After adding training words, " .. self.word_embedding.embeddings:size(1) .. " words")
  end

  self.model = StructuredEntailmentModel{word_emb = self.word_embedding,
                                         repr_dim = args.dim,
                                         num_relations = self.data.num_relations,
                                         learning_rate = args.learning_rate,
                                         batch_size = args.batch_size,
                                         dropout = args.dropout,
                                         interactive = true,
                                         words_from_embbedding = words_from_embedding,
                                         update_oov_only = args.oovonly,
                                         hiddenrel = args.hiddenrel,
                                         dataset = self.data,
                                         verbose = self.verbose}
end


function Trainer:train()
  local best_train_acc, best_dev_acc = 0, 0
  local train = self.data.train

  local profiler = SimpleProfiler()

  for i = 1, args.epochs do
    if self.verbose then printerr("Starting epoch " .. i) end

    profiler:reset()
    profiler:start("train")
    local train_info = self.model:train(train)
    profiler:pause("train")

    profiler:start("dev")
    local dev_info = self.model:evaluate(self.data.dev)

    profiler:pause("dev")

    local best_train_suffix, best_dev_suffix = "", ""
    if best_train_acc < train_info["acc"] then
      best_train_acc = train_info["acc"]
      best_train_suffix = "+"
    end
    if best_dev_acc < dev_info["acc"] then
      best_dev_acc = dev_info["acc"]
      best_dev_suffix = "+"
    end


    printerr(string.format("At epoch %d, train %.2fs loss %f acc %f%s dev %.2fs acc %f%s",
                           i, profiler:get_time("train"),
                           train_info["loss"], train_info["acc"], best_train_suffix,
                           profiler:get_time("dev"), dev_info["acc"], best_dev_suffix))

    if self.dump ~= "nil" then
      local filename = string.format("%s.%d.t7", self.dump, i)
      printerr("saving weights to ".. filename)
      torch.save(filename, self.model.params)
    end

  end
end


local t = Trainer()
if args.eval ~= "nil" then
  printerr("loading weights from ".. args.eval)
  local loaded = torch.load(args.eval)
  print("loaded params size", getTensorSize(loaded))
  t.model.params:copy(loaded)
  local eval_info = t.model:evaluate(t.data.dev)
  printerr(string.format("dev acc %f", eval_info["acc"]))
else
  t:train()
end
