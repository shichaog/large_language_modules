## About

This is a series implementation of GPT, from translation example to LLM examples.

The translation is based on a toy example implementation of paper《Attention is all you need》https://arxiv.org/abs/1706.03762.

The LLM currently is based on Llama-2, about SFT and Chinese Llama

In this repo I implement an english to french translation transformer in pytorch.

## environment
My environment is venv build by my pycharm python 3.7.16 venv and all requirements add install by pycharm according to requirements.txt.
you need install spacy language model in terminal by following shell commands:
```commandline
#python3 -m spacy download en_core_web_sm
#python3 -m spacy download fr_core_news_sm
```

## Dataset

The original data is stored in txt format in data directory, you can check each pair in en_to_fr.csv file.

## model and training process
model and training data preprocessing are all in train_translation_model.py file.
Since I already write a CSDN blog about detail of the implementation and there many references resources,
So I'm not explain each line of the codes here.

I hope this repo can help you much. Good luck.

## LlaMA-2 fine-tune 
202308 Update：
ADD LlaMA-2 fine-tune using single T4 GPU by colab.
