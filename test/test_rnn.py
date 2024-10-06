import unittest
import numpy as np
import torch 
import torch.nn as nn
from rnn import RNN,CrossEntropyLoss
from examples.names_classification import TensorHelper
from helper import all_categories,mock_data,RNNTorch,all_letters


class Test(unittest.TestCase):
    
    def test_rnn_gradient_comparison(self):
        #hyperparams
        hidden_size=50;epochs=1;n_letters=57

        tensor_helper=TensorHelper(all_categories=all_categories,n_letters=n_letters,all_letters=all_letters)
        #torch rnn 
        rnn_torch = RNNTorch(n_letters, hidden_size, len(all_categories))
        loss = nn.CrossEntropyLoss()
        #numpy rnn 
        rnn_numpy=RNN(n_letters=n_letters,hidden_size=hidden_size,learning_rate=0,target_length=len(all_categories))
        cross_entropy=CrossEntropyLoss(all_categories=all_categories)

        #copy the weights from torch to numpy 
        rnn_numpy.Wxh=rnn_torch.i2h.weight.T.detach().clone().numpy()
        rnn_numpy.Whh=rnn_torch.h2h.weight.T.detach().clone().numpy()
        rnn_numpy.Why=rnn_torch.h2o.weight.T.detach().clone().numpy()

        rnn_numpy.bx=rnn_torch.i2h.bias.detach().clone().numpy().reshape(1,-1)
        rnn_numpy.bh=rnn_torch.h2h.bias.detach().clone().numpy().reshape(1,-1)
        rnn_numpy.by=rnn_torch.h2o.bias.detach().clone().numpy().reshape(1,-1)

        # torch rnn forward/backward pass/gradient descent:
        for epoch in range(epochs):
            for name,label in mock_data.items():
                X=torch.from_numpy(tensor_helper.line_to_tensor(name)).to(torch.float32)
                Y=torch.from_numpy(tensor_helper.label_to_onehot(label)).to(torch.float32)
                
                hidden = rnn_torch.initHidden()
                rnn_torch.zero_grad()
                #forward pass
                for i in range(X.shape[0]):
                    output,hidden=rnn_torch(X[i],hidden)
                lossi = loss(output, Y.argmax(dim=1))
                
                #backward pass
                lossi.backward()
           
        # numpy rnn forward/backward pass/gradient descent:
        for epoch in range(epochs):
            for name,label in mock_data.items():
                #data preparation 
                X=tensor_helper.line_to_tensor(name)
                Y=tensor_helper.label_to_onehot(label)

                # forward pass
                logits,hprev=rnn_numpy.forward(X,rnn_numpy.init_hidden())
                probs,loss=cross_entropy.forward(logits,Y)
                #backward pass
                dy=cross_entropy.backward(probs,Y)
                dWxh, dWhh, dWhy, dbh, dby, dbx=rnn_numpy.backward(dy,X)
      
      
        self.assertTrue(np.allclose(dWxh,rnn_torch.i2h.weight.grad.T.numpy(),atol=0.01),'dWxh is not correct')
        self.assertTrue(np.allclose(dWhh,rnn_torch.h2h.weight.grad.T.numpy(),atol=0.01),'dWhh is not correct')
        self.assertTrue(np.allclose(dWhy,rnn_torch.h2o.weight.grad.T.numpy(),atol=0.01),'dWhy is not correct')
        self.assertTrue(np.allclose(dbx,rnn_torch.i2h.bias.grad.numpy(),atol=0.1),'dbx is not correct')
        self.assertTrue(np.allclose(dbh,rnn_torch.h2h.bias.grad.numpy(),atol=0.1),'dbh is not correct')
        self.assertTrue(np.allclose(dby,rnn_torch.h2o.bias.grad.numpy(),atol=0.1),'dby is not correct')
            

if __name__ == '__main__':
    unittest.main()