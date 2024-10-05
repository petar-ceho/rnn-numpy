import numpy as np

# this code is borrowed from the pytorch rnn tutorial notebook :)
class TensorHelper:
    
    def __init__(self,all_categories,n_letters,all_letters):
        self.all_categories=all_categories
        self.n_letters=n_letters
        self.all_letters=all_letters

    def label_to_onehot(self,label):
        one_hot=np.zeros((1,len(self.all_categories)))
        one_hot[0][self.all_categories.index(label)]+=1
        return one_hot

    def letter_to_index(self,letter):
        return self.all_letters.find(letter)

    def letter_to_tensor(self,letter):
        tensor=np.zeros((1,self.n_letters))
        tensor[0][self.letter_to_index(letter)]=1
        return tensor

    def line_to_tensor(self,line):
        tensor=np.zeros((len(line),1,self.n_letters))
        for li,letter in enumerate(line): 
            tensor[li][0][self.letter_to_index(letter)]=1
        return tensor
    




if __name__ == '__main__':
    
    



    
    pass





