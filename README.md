# Many-to-One Recurrent Neural Network (RNN) in NumPy

This repository implements a many-to-one Recurrent Neural Network (RNN) with Softmax and Categorical cross entropy using NumPy. Key features include:

- **[Variational Dropout](https://arxiv.org/abs/1512.05287)**: Applied to the RNN layers to improve generalization and avoid overfitting.
- **[Gradient Norm Clipping](https://arxiv.org/abs/1211.5063)**: Ensures stable training by limiting the norm of the gradients.
- **L1 Regularization**: Adds a penalty proportional to the absolute value of the weights to the loss function, promoting sparsity in the model.

- **L2 Regularization**: Adds a penalty proportional to the square of the weights to the loss function, helping to prevent overfitting by discouraging large weights.


## Acknowledgments

A significant portion of the RNN implementation was inspired by [Karpathy's RNN implementation](https://gist.github.com/karpathy/d4dee566867f8291f086). 


## Examples: Name Classification

This repository includes an example of name classification inspired by the PyTorch tutorial on character-level RNN classification. You can refer to the original tutorial for more details: [PyTorch Char-RNN Name Classification Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

Download the dataset put it in the data directory.  


## Unit Testing
Unit tests for this implementation compare the backward pass of the custom RNN in NumPy with the backward pass of PyTorch's RNN implementation. By comparing the gradients between PyTorch and NumPy.





## Training Results

### Configuration 1: 
- **Hidden Size**: 100  
- **Learning Rate**: 0.005  
- **Epochs**: 50  

#### Loss:
- **mean Training Loss (after 701,150 iterations)**: 0.6441  
- **mean Dev Loss (after 1,838 iterations)**: 3.2547  
- **mean Test Loss (after 1,865 iterations)**: 3.2738  

---

### Configuration 2: 
- **Hidden Size**: 50  
- **Learning Rate**: 0.001  
- **Epochs**: 50  

#### Loss:
- **mean Training Loss (after 701,150 iterations)**: 0.9307  
- **mean Dev Loss (after 1,838 iterations)**: 1.7544  
- **mean Test Loss (after 1,865 iterations)**: 1.6523  

---

### Configuration 3: 
- **L1 + L2 Regularization**

#### Loss:
- **mean Training Loss (after 701,150 iterations)**: 1.2028  
- **mean Dev Loss (after 1,838 iterations)**: 1.6978  
- **mean Test Loss (after 1,865 iterations)**: 1.5477  

---

### Configuration 4: 
- **With Variational Dropouts** (Gradient clipping needed due to exploding gradients)
- **Dropout Rate**: 0.4  
- **Max Gradient Norm**: 5.0  

#### Loss:
- **mean Training Loss (after 14,023 iterations)**: 1.1252  
- **mean Dev Loss (after 1,838 iterations)**: 1.3728  
- **mean Test Loss (after 1,865 iterations)**: 1.2689  



