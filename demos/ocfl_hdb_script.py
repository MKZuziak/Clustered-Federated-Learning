import os
import copy
from functools import partial
import pickle

from torch import optim
import timm
import datasets
from sklearn.cluster import HDBSCAN

from FedCL.model.federated_model import FederatedModel
from FedCL.node.federated_node import FederatedNode
from FedCL.simulation.simulation import Simulation
from FedCL.aggregators.fedopt_aggregator import Fedopt_Optimizer
from FedCL.files.archive import create_archive

def ocfl_hdb_script():
    # Defining global variables
    ROOT_PATH = os.getcwd() # ROOT PATH for all the results
    ARCHIVE_NAME = "OCFL_HDB_DEMO" # Name of the archives
    NET_ARCHITECTURE = timm.create_model('mobilenetv2_035', num_classes=10, pretrained=False, in_chans=1) # Net architecture for solving our task locally
    NUMBER_OF_CLIENTS=10 # Number of clients that will participate in the simulation.
    MIN_CLUSTER_SIZE = 3

    # Step 1: Create the direcory for storing results
    (
        metrics_savepath, 
        nodes_models_savepath, 
        orchestrator_model_savepath,
        clustering_root,
        sim_matrices_savepath,
        cluster_structure_savepath,
        ) = create_archive(
            path=ROOT_PATH,
            archive_name=ARCHIVE_NAME
        )
        
    # Step 2: Load the dataset
    dataset = datasets.load_dataset('ylecun/mnist')
    orchestrator_data = dataset['test'] # Using test-split as an orchestrator data
    nodes_data = [dataset['train'],dataset['test']] # Using train and test splits as nodes data.

    # Step 3: Defining the (local) optimizer's architecture and hyperparameters
    optimizer_architecture = partial(optim.SGD, lr=0.01) # Use functools.partial to create new function with partial application

    # Step 4: Define the model template
    model_tempate = FederatedModel(
        net=NET_ARCHITECTURE,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    ) # This will be a blueprint for creating object containing net architecture + individual dataloaders + helping methods 
    # for all the models (including orchestrators' model)

    # Step 5: Define the FederatedNode blueprint
    node_template = FederatedNode()
    # This will be a blueprint for a FederatedNode object that will be copied across all the nodes.

    # Step 6: Define the (central optimizer)
    fedopt_aggregator = Fedopt_Optimizer()
    # This will be a central optimizer used by Dthe DDorchestrator

    # Step 7: Define the simulation instance
    simulation_instace = Simulation(
        model_template=model_tempate,
        node_template=node_template
        )

    # Step 8: Attach the orchestrator dataset
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrator_data)

    # Step 9: Attach the node dataset
    simulation_instace.attach_node_model({
        node: copy.deepcopy(nodes_data) for node in range(NUMBER_OF_CLIENTS)
    })

    clustering_algorithm = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='precomputed')
    # Step 10: Initiate the simulation (use .training_protocol_baseline)
    simulation_instace.training_protocol_energy_oneshot(
        iterations=50,
        sample_size=NUMBER_OF_CLIENTS,
        local_epochs=3,
        aggrgator=fedopt_aggregator,
        learning_rate=1.0,
        clustering_algorithm=clustering_algorithm,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath,
        sim_matrices_savepath=sim_matrices_savepath,
        cluster_structure_savepath=cluster_structure_savepath
        )

if __name__ == "__main__":
    ocfl_hdb_script()