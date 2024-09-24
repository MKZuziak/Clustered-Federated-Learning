# What is Federated Clustered Learning (FCL)

FCL was projected to be a fully adjustable code base that allows its user to implement various algorithms for clustering clients in Federated Learning. Its components are fully modifiable building blocks that can be adjusted to your needs in a few lines of code. FCL was designed to be a minimalist Python boilerplate implementing your own clustering algorithms and comparing it with the state-of-the-art baseline. It does not aim to work as a Federated framework for actual communication between the devices, as there are already frameworks for doing that, e.g. [Flower](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html) or part of [PySyft](https://github.com/OpenMined/PySyft). However, the niche of a well-documented and easily adjustable framework for running clustering simulations in federated scenario is still open. This library has been created to be an integral part of the paper by M.K.Zuziak, R. Pellungrini and S. Rinzivillo: One-Shot Clustering for Federated Learning. It additionally implement two other clustering algorithms from papers, namely:

- F. Sattler, K.R. Muller and W. Samek: Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints, and
- C. Briggs, Z.Fan and P. Andras: Federated learning with hierarchical clustering of local updates to improve training on non-IID data

The modular and well-documented structure of the library allows for an easy implementation of other various clustering algorithms for testing purposes.

# How to use FCL?

As FCL was designed to be an easy-to-use boilerplate, it comes in several different ways to use it.
**Python pip**
As FCL is registered in the Python Package Index (PyPI), it can be downloaded using the pip install command. Simply run:
``pip install FCL``
**Template Cloning**
You can also clone the the repository using git and use it with poetry framework. The package is built with the help of [Poetry](https://python-poetry.org/). To install the project together with a virtual environment, navigate to the main folder and run:
``poetry install``.
The virtual environment will be installed together with the required dependencies.

## Project layout

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Examples](examples.md)
