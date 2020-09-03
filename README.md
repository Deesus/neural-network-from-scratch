# Simple Neural Network from Scratch

### Features:
- Custom initialization: He, Xavier
- Custom hyperparameters: learning rate, custom layers and layer size, number of iterations
- Custom regularization: L2 regularization, dropout
- Plot cost-iteration graph

### Notes:
- The duplicate `.py` files (e.g. `nn_binary_classification.py`) are Jupyter pairings -- used for diffing Jupyter Notebook changes via [Jupytext](https://github.com/mwouts/jupytext). Normally, we'd version control only the `.py` files and ignore the `.ipynb` pairings; however, for quick viewing on GitHub and general convenience, I'm keeping both file extensions.

### TODO:
- [ ] Use Kwargs** instead of Args** for class constructor
- [ ] Add API documentation
- [ ] Replace `nn_utils` with custom functions
- [ ] Add method docstrings
- Add features:
    - [ ] Mini-batch gradient descent
    - [ ] Batch norm
    - [ ] Algorithms: momentum, RMS Prop, Adam

### License:
Copyright 2020 Dee Reddy. BSD-2 License.
