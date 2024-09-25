import os
from functools import partial
import pickle

from torch import optim
from sklearn.cluster import KMeans

from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork
from FedCL.model.federated_model import FederatedModel
from FedCL.node.federated_node import FederatedNode
from FedCL.simulation.simulation import Simulation
from FedCL.aggregators.fedopt_aggregator import Fedopt_Optimizer
from FedCL.files.archive import create_archive

def integration_test():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath,
     clustering_root,
     sim_matrices_savepath,
     cluster_structure_savepath,
     ) = create_archive(os.getcwd())
    
    
    with open(f'/home/mzuziak/snap/snapd-desktop-integration/83/Documents/FedCL/tests/test_props/clustered_dataset/FMNIST_30_dataset_pointers', 'rb') as file:
        data = pickle.load(file)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    
    net_architecture = NeuralNetwork()
    optimizer_architecture = partial(optim.SGD, lr=0.001)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
    
    simulation_instace = Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrator_data)
    simulation_instace.attach_node_model({
        node: nodes_data[node] for node in range(30)
    })
    
    clustering_algorithm = KMeans(
        n_clusters=2
    )
    
    simulation_instace.training_protocol(
        iterations=5,
        sample_size=30,
        local_epochs=1,
        aggrgator=fed_avg_aggregator,
        learning_rate=1.0,
        clustering_algorithm=clustering_algorithm,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath,
        sim_matrices_savepath=sim_matrices_savepath,
        cluster_structure_savepath=cluster_structure_savepath
    )


if __name__ == "__main__":
    integration_test()