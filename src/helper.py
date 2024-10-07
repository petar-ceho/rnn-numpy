import numpy as np
from io import open
import glob
import unicodedata
import string

# this code is borrowed from the pytorch rnn tutorial from here,just added numpy instead of torch
# check: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
all_letters = string.ascii_letters + " .,;'" 
n_letters = len(all_letters)

class TensorHelper:
    
    def __init__(self,all_categories,n_letters,all_letters):
        self.all_categories=all_categories
        self.n_letters=n_letters
        self.all_letters=all_letters
 
    def label_to_onehot(self,label):
        one_hot=np.zeros((1,len(self.all_categories)))
        one_hot[0][self.all_categories.index(label)]+=1  
        return one_hot # shape [1,all_categories] one-hot

    def letter_to_index(self,letter):
        return self.all_letters.find(letter)

    def letter_to_tensor(self,letter):
        tensor=np.zeros((1,self.n_letters))
        tensor[0][self.letter_to_index(letter)]=1
        return tensor #shape [1,n_letters] one-hot 

    def line_to_tensor(self,line):
        tensor=np.zeros((len(line),1,self.n_letters))
        for li,letter in enumerate(line): 
            tensor[li][0][self.letter_to_index(letter)]=1
        return tensor #shape [n,1,n_letters]
    

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]
