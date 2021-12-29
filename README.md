## Empirical Dynamic Modeling (EDM) Jupyter Notebook
---
A Jupyter notebook GUI front-end for the [pyEDM](https://github.com/SugiharaLab/pyEDM) package. An introduction to EDM with documentation is avilable [online](https://sugiharalab.github.io/EDM_Documentation/ "EDM Docs"). The pyEDM package documentation is in the [API docs](https://github.com/SugiharaLab/pyEDM/blob/master/doc/pyEDM.pdf "pyEDM API"). The EDM packages are developed and maintained by the [Sugihara Lab](http://deepeco.ucsd.edu/).

Functionality includes:
* Simplex projection ([Sugihara and May 1990](https://www.nature.com/articles/344734a0))
* Sequential Locally Weighted Global Linear Maps (S-map) ([Sugihara 1994](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1994.0106))
* Multivariate embeddings ([Dixon et. al. 1999](https://science.sciencemag.org/content/283/5407/1528))
* Convergent cross mapping ([Sugihara et. al. 2012](https://science.sciencemag.org/content/338/6106/496))
* Multiview embedding ([Ye and Sugihara 2016](https://science.sciencemag.org/content/353/6302/922))

---
## Installation

### pyEDM Python Package
pyEDM is hosted on the Python Package Index respository (PyPI) at [pyEDM](https://pypi.org/project/pyEDM/).

It can be installed from the command line using the Python pip module: `python -m pip install pyEDM`.

### Jupyter notebook
Download the jpyEDM source.  
Start Jupyter notebook.  
Open `"jpyEDM/notebooks/pyEDM Version 0.ipynb"`.

---
## Introduction
A brief [video presentation](https://github.com/SugiharaLab/jpyEDM/blob/master/doc/jpyEDM_Introduction-2021-12-29.mp4).

---
## Screenshot
![picture alt](https://github.com/SugiharaLab/jpyEDM/blob/master/doc/jpyEDM-CCM-Screen.png "CCM Lorenz 5D")

---
### References
Sugihara G. and May R. 1990.  Nonlinear forecasting as a way of distinguishing 
chaos from measurement error in time series. Nature, 344:734–741.

Sugihara G. 1994. Nonlinear forecasting for the classification of natural 
time series. Philosophical Transactions: Physical Sciences and 
Engineering, 348 (1688) : 477–495.

Dixon, P. A., M. Milicich, and G. Sugihara, 1999. Episodic fluctuations in larval supply. Science 283:1528–1530.

Sugihara G., May R., Ye H., Hsieh C., Deyle E., Fogarty M., Munch S., 2012.
Detecting Causality in Complex Ecosystems. Science 338:496-500.

Ye H., and G. Sugihara, 2016. Information leverage in interconnected 
ecosystems: Overcoming the curse of dimensionality. Science 353:922–925.
