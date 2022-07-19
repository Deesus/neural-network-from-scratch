# Deep Neural Networks from Scratch

A mini "framework" for deep neural networks, written using only NumPy.
Check out the [interactive notebook](https://github.com/Deesus/neural-network-from-scratch/blob/master/nn_classification.ipynb). Currently, only binary classification and multi-class classification problems are supported.

### Features:
- Backpropagation implementation
- Custom initialization: He, Xavier
- Custom hyperparameters: learning rate, custom layers and layer size, number of iterations
- Custom regularization: L2 regularization, dropout
- Gradient Descent Optimizers: Momentum, Adam, RMSProp

### Notes:
- The duplicate `.py` files (e.g. `nn_binary_classification.py`) are Jupyter pairings -- used for diffing Jupyter Notebook changes via [Jupytext](https://github.com/mwouts/jupytext). Normally, we'd version control only the `.py` files and ignore the `.ipynb` pairings; however, for quick viewing on GitHub and general convenience, I'm keeping both file extensions.

### TODO:
- [ ] Add API documentation
- [ ] Replace `nn_utils` with custom functions
- [ ] Add method docstrings
- [ ] Add features:
    - [ ] Mini-batch gradient descent
    - [ ] Batch norm
    - [ ] Softmax

### License:
Copyright 2020-2021 Deepankara Reddy. BSD-2 License.
