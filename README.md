# Many-to-One Recurrent Neural Network (RNN) in NumPy

This repository implements a many-to-one Recurrent Neural Network (RNN) with Softmax and Categorical cross entropy using NumPy. Key features include:

- **[Variational Dropout](https://arxiv.org/abs/1512.05287)**: Applied to the RNN layers to improve generalization and avoid overfitting.
- **[Gradient Norm Clipping](https://arxiv.org/abs/1211.5063)**: Ensures stable training by limiting the norm of the gradients.

## Acknowledgments

A significant portion of the RNN implementation was inspired by [Karpathy's RNN implementation](https://gist.github.com/karpathy/d4dee566867f8291f086). 



## Examples: Name Classification

This repository includes an example of name classification inspired by the PyTorch tutorial on character-level RNN classification. You can refer to the original tutorial for more details: [PyTorch Char-RNN Name Classification Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).



## Unit Testing
Unit tests for this implementation compare the backward pass of the custom RNN in NumPy with the backward pass of PyTorch's RNN implementation. By comparing the gradients between PyTorch and NumPy, we can verify that the custom model behaves as expected(check test directory).







