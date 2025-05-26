Before intiating first read this. i assuming you are reading this and hada quite experince in python. lets not get muddy.

things you need to have knowledge to work in this codebase:

1.pytorch(not entirly, gain wahats only nedded).
2.argparse
3.Math,Numpy module
4.Huggingface & Tokenizer library(e.g. tiktoken, huggingface autotokenizer, sentenciepiece if nedded).

i have already putted a torch-example.ipynb file for your better undesrsatnding thus you dont need to stress over on documentations.
The Fira-v01.ipynb is the working code of my kaggle its the latest works of mine except the MoE architecture.
I also attached some reference paper in the reference directory where my idea came from and how things work mathmetically.
its more like a book for documentations.

TODO:

our first goal is to build and train the tokenizer.json file. my latest tokenizer works is in the tokenizer.py file.
and also its your first task. its very importent beacuse with out it we have to relay on gpt2 tokenizer or other tokenizer library.
which is not a good practice for a company and it cames with cons.

2nd modify the code with the MoE to implement our own Tokenizer and train our first model. then we can pretrain it on our own.
And we can replace it with the pretrain GPT2HEADMODEL.

chatbot.py is use for inference from the model.

i have attach and image of my datasets in kaggle. you can find all of them in kaggle and make sure to use kaggle for training and allof the codebase.

datasets:

for tokenizer training training:

https://www.kaggle.com/datasets/himonsarkar/openwebtext-dataset

for training:

https://www.kaggle.com/datasets/vadimkurochkin/openwebtext

the new traing code with MoE still need modification from the train.py like using the dataset as in .bin and many others.
