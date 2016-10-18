# Textual Entailment with Structured Attentions and Composition

This repository contains an implementation of the structured attention and compositon model for textual entailment described in the paper [Textual Entailment with Structured Attentions and Composition](#).

#### Required Dependencies

1. Torch7
2. Torch [rnn](https://github.com/Element-Research/rnn) library

#### Training

To train on the provided sample data and saving, you can simply run:

```
th trainer.lua --dump model_file
```

You can find the training parameters and their descriptions in file `trainer.lua`. 

#### Evaluating

To evaluate the trained model on the dev set, you can run:

```
th trainer.lua --eval model_file
```
