## Jupytext Commands

See https://github.com/mwouts/jupytext/blob/master/docs/faq.md
Also see https://jupytext.readthedocs.io/en/latest/

Once a `.ipynb` is **paired** via jupytext, you have a couple of options:
1. You can work in the Notebook (`.ipynb`) and after saving, delete the paired `.py` file then run **sync**, which will create an updated `.py` file.

### Primary commands:
- Sync .ipynb:
```bash
jupytext --sync NOTEBOOK_NAME.ipynb
```

- pair a notebook:
```bash
jupytext --set-formats ipynb,py NOTEBOOK_NAME.ipynb
```

- pair multiple notebooks:
```bash
jupytext --set-formats ipynb,py *.ipynb
```

### Misc. commands: 
```bash
jupytext --to py NOTEBOOK_NAME.ipynb                 # convert notebook.ipynb to a .py file
jupytext --to notebook NOTEBOOK_NAME.py              # convert notebook.py to an .ipynb file with no outputs
jupytext --to notebook --execute NOTEBOOK_NAME.md    # convert notebook.md to an .ipynb file and run it
jupytext --update --to notebook NOTEBOOK_NAME.py     # update the input cells in the .ipynb file and preserve outputs and metadata
jupytext --set-formats ipynb,py NOTEBOOK_NAME.ipynb  # Turn notebook.ipynb into a paired ipynb/py notebook
jupytext --sync NOTEBOOK_NAME.ipynb                  # Update all paired representations of notebook.ipynb
```
