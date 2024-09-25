# What is Federated Clustered Learning (FCL)

Federated Clustered Learning (FCL) was designed to be a fully adjustable code base that allows its users to implement various algorithms for clustering clients in Federated Learning. Its components are fully modifiable building blocks that can be fitted to your needs in a few lines of code. FCL was designed to be a minimalist Python boilerplate that allows the community to experiment with various clustering algorithms in federated scenarios. It does not aim to work as a Federated framework for actual communication between the devices, as there are already frameworks for doing that, e.g. [Flower](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html) or part of [PySyft](https://github.com/OpenMined/PySyft). However, the niche of a well-documented and easily adjustable framework for running clustering simulations in federated scenarios is still open. This library has been created to be an integral part of the paper by M.K.Zuziak, R. Pellungrini and S. Rinzivillo: One-Shot Clustering for Federated Learning. It additionally implements two other clustering algorithms from papers, namely:
- F. Sattler, K.R. Muller and W. Samek: Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints, and
- C. Briggs, Z.Fan and P. Andras: Federated learning with hierarchical clustering of local updates to improve training on non-IID data

# How to use FCL?
As FCL was designed to be an easy-to-use boilerplate, it comes in several different ways to use it.
## PIP
The FCL library is registered in the Python Package Index (PyPI) under the name of FedCL and it can be downloaded using the pip install command. Simply run:
``pip install fedcl``
While using the pip-installed FedCL, it is possible to import modules directly, as illustrated in the **Examples** section/
## Poetry Install
The FCL library can also cloned and used together with the [Poetry Framework](https://python-poetry.org/). 
It requires cloning the repository directly from the GitHub. To install the project together with a virtual environment, navigate to the main folder and run the following command:
``poetry install``.
The virtual environment will be installed together with the required dependencies.
To use the FCL modules together with the poetry environment, you should activate the poetry shell at the beginning of each session by executing the command:
``poetry shell``
Alternatively, it is possible to execute scripts without directly entering the poetry shell. For details, [consult the original documentation](https://python-poetry.org/).
