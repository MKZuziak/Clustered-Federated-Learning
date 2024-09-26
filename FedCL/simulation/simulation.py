import os
import copy
from collections import OrderedDict
from typing import Any
from multiprocessing import Pool

import pickle
import numpy as np
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering

from FedCL.model.federated_model import FederatedModel
from FedCL.node.federated_node import FederatedNode
from FedCL.operations.orchestrations import train_nodes, sample_nodes
from FedCL.aggregators.aggregator import Aggregator
from FedCL.aggregators.distances import calculate_cosine_similarity, calculate_cosine_dist, compute_max_update_norm, compute_mean_update_norm
from FedCL.aggregators.temperature import calculate_temperature
from FedCL.operations.evaluations import evaluate_model, automatic_node_evaluation, automatic_generalizability_evaluation
from FedCL.files.handlers import save_nested_dict_ascsv
from FedCL.files.loggers import orchestrator_logger
from FedCL.utils.computations import average_of_weigts
from FedCL.utils.select_gradients import select_gradients
from FedCL.data_structures.cluster_sturcutre import Cluster_Structure
from FedCL.utils.splitters import chunker_dict


# set_start_method set to 'spawn' to ensure compatibility across platforms.
# from multiprocessing import set_start_method
# set_start_method("spawn", force=True)
# Setting up the orchestrator logger
orchestrator_logger = orchestrator_logger()


class Simulation():
    """Simulation class representing a generic simulation type.
        
        Attributes
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that is uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that is used to simulate nodes.
        data : dict
            Local data used for the training in a dictionary format, mapping each client to its respective dataset.
        """
    
    
    def __init__(
        self, 
        model_template: FederatedModel,
        node_template: FederatedNode,
        seed: int = 42,
        **kwargs
        ) -> None:
        """Creating simulation instant requires providing an already created instance of model template
        and node template. Those instances then will be copied n-times to create n different nodes, each
        with a different dataset. Additionally, a data for local nodes should be passed in form of a dictionary,
        maping dataset to each respective client.

        Parameters
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that will be uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that will be used to simulate nodes.
        seed : int,
            Seed for the simulation, default to 42
        **kwargs : dict, optional
            Extra arguments to enable selected features of the Orchestrator.
            passing full_debug to **kwargs, allow to enter a full debug mode.

        Returns
        -------
        None
        """
        self.model_template = model_template
        self.node_template = node_template
        self.network = {}
        self.orchestrator_model = None
        self.generator = np.random.default_rng(seed=seed)
        self.current_clusters = OrderedDict()
        self.flattened_model_length = 0
        self.clusters_namespace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', "K",
                                   "L", "M", "N", "O", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
    
    def attach_orchestrator_model(
        self,
        orchestrator_data: Any
        ) -> None:
        """Attaches model of the orchestrator that is saved as an instance attribute.
        
        Parameters
        ----------
        orchestrator_data: Any 
            Orchestrator data that should be attached to the orchestrator model.
        
        Returns
        -------
        None
        """
        self.orchestrator_model = copy.deepcopy(self.model_template)
        self.orchestrator_model.attach_dataset_id(local_dataset=[orchestrator_data],
                                                  node_name='orchestrator',
                                                  only_test=True)
    
    
    def attach_node_model(
        self,
        nodes_data: dict
        ) -> None:
        """Attaches models of the nodes to the simulation instance.
        
        Parameters
        ----------
        orchestrator_data: Any
            Orchestrator data that should be attached to nodes models.
        Returns
        -------
        None
        """
        for node_id, data in nodes_data.items():
            self.network[node_id] = copy.deepcopy(self.node_template)
            self.network[node_id].connect_data_id(node_id = node_id,
                                                  model = copy.deepcopy(self.model_template),
                                                  data=data)
        # Calculting the length of the model -> for creating similarity matrix
        flattened_model = []
        model = self.network[next(iter(self.network))].model
        for value in model.get_weights().values():
            flattened_model.extend(value.flatten().detach().numpy().tolist()) 
        self.flattened_model_length = len(flattened_model)
    

    def train_epoch(
        self,
        sampled_nodes: dict[int: FederatedNode],
        iteration: int,
        local_epochs: int, 
        mode: str = 'weights',
        save_model: bool = False,
        save_path: str = None,
        batch_job = False
        ) -> tuple[dict[int, int, float, float, list, list], dict[int, OrderedDict]]:
        """Performs one training round of a federated learning. Returns training
        results upon completion.
        
        Parameters
        ----------
        samples_nodes: dict[int: FederatedNode]
            Dictionary containing sampled Federated Nodes
        iteration: int
            Current global iteration of the training process
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        mode: str (default to 'weights')
            Mode = 'weights': Node will return model's weights.
            Mode = 'gradients': Node will return model's gradients.
        save_model: bool (default to False)
            Boolean flag to save a model.
        save_path: str (defualt to None)
            Path object used to save a model

        Returns
        -------
        tuple[dict[int, int, float, float, list, list], dict[int, OrderedDict]]
        """
        training_results = {}
        weights = {}
        if batch_job:
            for batch in chunker_dict(sampled_nodes, size=1):
                with Pool(len(batch)) as pool:
                    results = [pool.apply_async(train_nodes, (node, iteration, local_epochs, mode, save_model, save_path)) for node in list(batch.values())]
                    for result in results:
                        node_id, model_weights, loss_list, accuracy_list = result.get()
                        weights[node_id] = model_weights
                        training_results[node_id] = {
                            "iteration": iteration,
                            "node_id": node_id,
                            "loss": loss_list[-1], 
                            "accuracy": accuracy_list[-1],
                            "full_loss": loss_list,
                            "full_accuracy": accuracy_list}
        else:
            with Pool(len(sampled_nodes)) as pool:
                results = [pool.apply_async(train_nodes, (node, iteration, local_epochs, mode, save_model, save_path)) for node in list(sampled_nodes.values())]
                for result in results:
                    node_id, model_weights, loss_list, accuracy_list = result.get()
                    weights[node_id] = model_weights
                    training_results[node_id] = {
                        "iteration": iteration,
                        "node_id": node_id,
                        "loss": loss_list[-1], 
                        "accuracy": accuracy_list[-1],
                        "full_loss": loss_list,
                        "full_accuracy": accuracy_list}
        return (training_results, weights)


    def no_clustering(
        self,
        iteration: int,
        gradients: OrderedDict,
        sim_mat_savepath: str,
        temperature_save_path: str,
    ):
        orchestrator_logger.info(f"[ITERATION {iteration}] CLUSTERING PROCEDURE: NO CLUSTERING")
        # Step I: Loop over all clusters of iteration = iteration
        loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iterating
        for cluster_signature in loop_current_clusters_keys:
            # Define a new name of 
            # cluster: signature+in_cluster_nodes
            cluster_name = f"{cluster_signature}" + "-".join((str(id) for id in self.current_clusters[cluster_signature].nodes))
            # Collect in-cluster gradients
            in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
            )
            # Creating matrix of appended gradients
            in_cluster_gradients_matrix = np.zeros(shape=(len(in_cluster_gradients.keys()), self.flattened_model_length))
            for pos, ind_gradients in enumerate(in_cluster_gradients.values()):
                # Flatenning model
                flattened_model = []
                for value in ind_gradients.values():
                    flattened_model.extend(value.flatten().detach().numpy().tolist())
                in_cluster_gradients_matrix[pos,:] = np.array(flattened_model)
            # Calculating Cosine Distance
            cosdist_mat = calculate_cosine_dist(in_cluster_gradients_matrix)
            # Save matrix of cosine distance
            with open(os.path.join(sim_mat_savepath, f'cluster_of_{cluster_name}_at_{iteration}'), 'wb') as file:
                pickle.dump(cosdist_mat, file)
            temperature = calculate_temperature(cosdist_mat)
            with open(os.path.join(temperature_save_path, 'clusters_temperature.csv'), 'a+', newline='\n') as file:
                file.write(f"{iteration},{cluster_name},{temperature}\n")


    def energy_oneshot(
        self,
        clustering_algorithm : cluster,
        last_temperature: float,
        iteration: int,
        gradients: OrderedDict,
        sim_mat_savepath: str,
        temperature_save_path: str,
        clustering_shot: str,
    ):
        orchestrator_logger.info(f"[ITERATION {iteration}] CLUSTERING PROCEDURE: ONESHOT CLUSTERING")
        # Step I: Loop over all clusters of iteration = iteration
        loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iteratin
        for cluster_signature in loop_current_clusters_keys:
            # Define a new name: signature+in_cluster_nodes
            cluster_name = f"{cluster_signature}" + "-".join((str(id) for id in self.current_clusters[cluster_signature].nodes))
            # Collect in-cluster gradients
            in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
            )
            # Creating matrix of appended gradients
            in_cluster_gradients_matrix = np.zeros(shape=(len(in_cluster_gradients.keys()), self.flattened_model_length))
            for pos, ind_gradients in enumerate(in_cluster_gradients.values()):
                # Flatenning model
                flattened_model = []
                for value in ind_gradients.values():
                    flattened_model.extend(value.flatten().detach().numpy().tolist())
                in_cluster_gradients_matrix[pos,:] = np.array(flattened_model)
            # Calculating Cosine Distance
            cosdist_mat = calculate_cosine_dist(in_cluster_gradients_matrix)
            # Save matrix of cosine distance
            with open(os.path.join(sim_mat_savepath, f'cluster_of_{cluster_name}_at_{iteration}'), 'wb') as file:
                pickle.dump(cosdist_mat, file)
            # Calculating matrix temperature           
            temperature = calculate_temperature(cosdist_mat)
            # Save matrix temperature
            with open(os.path.join(temperature_save_path, 'clusters_temperature.csv'), 'a+', newline='\n') as file:
                file.write(f"{iteration},{cluster_signature},{temperature}\n")
            orchestrator_logger.info(f"[ITERATION {iteration}] Last Temperature: {last_temperature}")
            orchestrator_logger.info(f"[ITERATION {iteration}] Current Temperature: {temperature}")
            # Step II: Check if clustering was already performed - if yes, skip the clustering part and return True
            if clustering_shot==False:
                # Step III: If temperature exceeds minimum perform clustering
                if temperature > last_temperature:
                    orchestrator_logger.info(f"[ITERATION {iteration}] Raise in temperature detected, performing clustering")
                    # Using clustering algorithm
                    clustering_algorithm.fit_predict(cosdist_mat)
                    new_clusters_info = []
                    # Adding children to the current clustering structure
                    self.all_clusters[cluster_signature].children = [] # Preparing to add children to the cluster
                    for label, new_signature in zip(np.unique(clustering_algorithm.labels_), self.clusters_namespace):
                        new_clusters_name = f"{cluster_signature}{new_signature}"
                        new_clusters_info.append(new_clusters_name)
                        self.current_clusters[new_clusters_name] = Cluster_Structure(
                            population=self.all_clusters[cluster_signature].nodes[np.where(clustering_algorithm.labels_==label)],
                            model=copy.deepcopy(self.all_clusters[cluster_signature].model),
                            iteration=iteration,
                            parent=self.current_clusters[cluster_signature])
                        self.all_clusters[cluster_signature].children.append(self.current_clusters[new_clusters_name])
                        self.all_clusters[new_clusters_name] = self.current_clusters[new_clusters_name]
                    # Adding old cluster to a set of all clusters
                    # Deleting old cluster from a set of current clusters
                    del self.current_clusters[cluster_signature]
                    orchestrator_logger.info(f"[ITERATION {iteration}] Cluster of {cluster_signature} was partitioned into {new_clusters_info}")
                    # Extending next iteration structure by the current children, returning True ('shot fired')
                    # Saving the cluster
                    return (True, None)
                else:
                    # Else returning False ('shot not fired')
                    # # if T <= T_min, we just carry on the cluster to the next entry.
                    return (False, temperature)
            else:
                # Do not change the clustering structure
                return (True, None)
    
    
    def briggs_hierarchical(
        self,
        iteration: int,
        gradients: OrderedDict,
        sim_mat_savepath: str,
        distance_threshold: float
    ):
        clustering_algorithm = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=distance_threshold,
                                                       linkage='average')
        orchestrator_logger.info(f"[ITERATION {iteration}] CLUSTERING PROCEDURE: BRIGGS CLUSTERING")
        # Step I: Loop over all clusters of iteration = iteration
        loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iterating
        for cluster_signature in loop_current_clusters_keys:
            cluster_name = f"{cluster_signature}" + "-".join((str(id) for id in self.current_clusters[cluster_signature].nodes))
            # Collect in-cluster gradients
            in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
            )
            # Creating matrix of appended gradients
            in_cluster_gradients_matrix = np.zeros(shape=(len(in_cluster_gradients.keys()), self.flattened_model_length))
            for pos, ind_gradients in enumerate(in_cluster_gradients.values()):
                # Flatenning model
                flattened_model = []
                for value in ind_gradients.values():
                    flattened_model.extend(value.flatten().detach().numpy().tolist())
                in_cluster_gradients_matrix[pos,:] = np.array(flattened_model)
            # Calculating Cosine Similarity
            cossim_mat = calculate_cosine_similarity(in_cluster_gradients_matrix)
            # Save matrix of cosine similarity
            with open(os.path.join(sim_mat_savepath, f'cluster_of_{cluster_name}_at_{iteration}'), 'wb') as file:
                pickle.dump(cossim_mat, file)
            # Step III: 
            # Using clustering algorithm
            clustering_algorithm.fit_predict(cossim_mat)
            new_clusters_info = []
            # Adding children to the current clustering structure
            self.all_clusters[cluster_signature].children = [] # Preparing to add children to the cluster
            for label, new_signature in zip(np.unique(clustering_algorithm.labels_), self.clusters_namespace):
                new_clusters_name = f"{cluster_signature}{new_signature}"
                new_clusters_info.append(new_clusters_name)
                self.current_clusters[new_clusters_name] = Cluster_Structure(
                    population=self.all_clusters[cluster_signature].nodes[np.where(clustering_algorithm.labels_==label)],
                    model=copy.deepcopy(self.all_clusters[cluster_signature].model),
                    iteration=iteration,
                    parent=self.current_clusters[cluster_signature])
                self.all_clusters[cluster_signature].children.append(self.current_clusters[new_clusters_name])
                self.all_clusters[new_clusters_name] = self.current_clusters[new_clusters_name]
            # Adding old cluster to a set of all clusters
            # Deleting old cluster from a set of current clusters
            del self.current_clusters[cluster_signature]
            orchestrator_logger.info(f"[ITERATION {iteration}] Cluster of {cluster_signature} was partitioned into {new_clusters_info}")
            # Extending next iteration structure by the current children, returning True ('shot fired')
            # Saving the cluster
            return (True, None)
    
    
    def sattler_clustering(
        self,
        iteration: int,
        gradients: OrderedDict,
        sim_mat_savepath: str,
        EPS_1:float,
        EPS_2:float,
        round_cooldown:int
    ):
        orchestrator_logger.info(f"[ITERATION {iteration}] CLUSTERING PROCEDURE: SATTLER CLUSTERING")
        # Step I: Loop over all clusters of iteration = iteration
        loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iteratin
        for cluster_signature in loop_current_clusters_keys:
            cluster_name = f"{cluster_signature}" + "-".join((str(id) for id in self.current_clusters[cluster_signature].nodes))
            # Collect in-cluster gradients
            in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
            )
            in_cluster_gradients_matrix = np.zeros(shape=(len(in_cluster_gradients.keys()), self.flattened_model_length))
            for pos, ind_gradients in enumerate(in_cluster_gradients.values()):
                # Flatenning model
                flattened_model = []
                for value in ind_gradients.values():
                    flattened_model.extend(value.flatten().detach().numpy().tolist())
                in_cluster_gradients_matrix[pos,:] = np.array(flattened_model)
            cossim_mat = calculate_cosine_similarity(in_cluster_gradients_matrix)
            with open(os.path.join(sim_mat_savepath, f'cluster_of_{cluster_name}_at_{iteration}'), 'wb') as file:
                pickle.dump(cossim_mat, file)
            max_norm = compute_max_update_norm(in_cluster_gradients_matrix)
            mean_norm = compute_mean_update_norm(in_cluster_gradients_matrix)
            orchestrator_logger.info(f"[ITERATION {iteration}] Mean norm: {mean_norm}, max norm: {max_norm}")
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(self.current_clusters[cluster_signature].nodes) > 2 and iteration > round_cooldown:
                orchestrator_logger.info(f"[ITERATION {iteration}] Clustering criterions met, perform clustering.")
                clustering_algorithm = AgglomerativeClustering(metric='precomputed', linkage ='complete').fit(-cossim_mat)
                new_clusters_info = []
                # Adding children to the current clustering structure
                self.all_clusters[cluster_signature].children = [] # Preparing to add children to the cluster
                for label, new_signature in zip(np.unique(clustering_algorithm.labels_), self.clusters_namespace):
                    new_clusters_name = f"{cluster_signature}{new_signature}"
                    new_clusters_info.append(new_clusters_name)
                    self.current_clusters[new_clusters_name] = Cluster_Structure(
                        population=self.all_clusters[cluster_signature].nodes[np.where(clustering_algorithm.labels_==label)],
                        model=copy.deepcopy(self.all_clusters[cluster_signature].model),
                        iteration=iteration,
                        parent=self.current_clusters[cluster_signature])
                    self.all_clusters[cluster_signature].children.append(self.current_clusters[new_clusters_name])
                    self.all_clusters[new_clusters_name] = self.current_clusters[new_clusters_name]
                    # Adding old cluster to a set of all clusters
                    # Deleting old cluster from a set of current clusters
                del self.current_clusters[cluster_signature]
                orchestrator_logger.info(f"[ITERATION {iteration}] Cluster of {cluster_signature} was partitioned into {new_clusters_info}")
            else:
                pass
    

    def training_protocol_energy_oneshot(
        self,
        iterations: int,
        sample_size: int,
        local_epochs: int,
        aggrgator: Aggregator,
        learning_rate: float,
        clustering_algorithm: cluster,
        metrics_savepath: str,
        nodes_models_savepath: str,
        orchestrator_models_savepath: str,
        sim_matrices_savepath: str,
        cluster_structure_savepath: str,
        skip_calculation: bool = True,
        batch_job : bool = False
        ) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedOpt algorithm - it averages the local gradients 
        and aggregates them using a selected optimizer.
        SOURCE: 

        Parameters
        ----------
        iterations: int
            Number of (global) iterations // epochs to train the models for.
        sample_size: int
            Size of the sample
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        aggregator: Aggregator
            Instance of the Aggregator object that will be used to aggregate the result each round
        learning_rate: float
            Learning rate to be used for optimization.
        metrics_savepath: str
            Path for saving the metrics
        nodes_models_savepath: str
            Path for saving the models in the .pt format.
        orchestrator_models_savepath: str
            Path for saving the orchestrator models.
        clustering_root_savepath: str
            Path for saving all the clustering results (general).
        sim_matrices_savepath: str
            Path for saving all the similarity matrices.
        cluster_structure_savepath: str
            Path for saving all the cluster structures.
        skip_calculation: bool
            If set to True, will caculate Cosine Distance / Similarity values only when clustering.
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        # Step I: Create first cluster (the whole population)
        self.current_clusters = {'A': Cluster_Structure(
            population=np.array([node for node in self.network]), # First cluster, all nodes belonging to it.
            model = copy.deepcopy(self.orchestrator_model.get_weights()), # Model for the first cluster, direct copy of the central orchestrator model.
            iteration=0 # Created During the First Iteration
        )}
        self.all_clusters = {'A': self.current_clusters['A']} # List of all clusters that were created during the simulation
        self.node_cluster_mapping = OrderedDict() # mapping between iteration: {node: cluster_id}
        
        # Step II: Setting the 'clustering shot' flag to False
        clustering_shot = False
        last_temperature = float('inf')
        
        # Step III: Looping over the iterations
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")            
            # Sampling nodes
            sampled_nodes = sample_nodes(
                nodes = self.network,
                sample_size = sample_size,
                generator = self.generator
            )
            
            # Step IV: Training nodes
            training_results, gradients = self.train_epoch(
                sampled_nodes=sampled_nodes,
                iteration=iteration,
                local_epochs=local_epochs,
                mode='gradients',
                save_model=True,
                save_path=nodes_models_savepath,
                batch_job=batch_job
            )
            # Preserving metrics of the training
            save_nested_dict_ascsv(
                data=training_results,
                save_path=os.path.join(metrics_savepath, 'training_metrics.csv'))
            # Testing nodes on the local dataset before the model update (only sampled nodes).
            automatic_node_evaluation(
                iteration=iteration,
                nodes=sampled_nodes,
                save_path=os.path.join(metrics_savepath, "before_update_metrics.csv"))
                        # Evaluating the generalizability of the local model
            automatic_generalizability_evaluation(
                iteration=iteration,
                nodes=self.network,
                general_model=copy.deepcopy(self.orchestrator_model),
                save_path=os.path.join(metrics_savepath, "after_update_generalizability.csv")
            )
            # If clustering was performed before and we want to skip saving cosdist mat
            if clustering_shot and skip_calculation:
                pass # Do not change the self.current_clusters architecture
            # Else perform a full procedure
            else:
                # Clustering
                clustering_shot, last_temperature = self.energy_oneshot(
                    clustering_algorithm = clustering_algorithm,
                    last_temperature = last_temperature,
                    iteration = iteration,
                    gradients = copy.deepcopy(gradients),
                    sim_mat_savepath = sim_matrices_savepath,
                    temperature_save_path = metrics_savepath,
                    clustering_shot = clustering_shot
                )
            
            self.node_cluster_mapping[iteration] = {}
            loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iterating # Copy obj for iterating
            # Step V: In-cluster aggregation
            for cluster_signature in loop_current_clusters_keys:
                # Collect in-cluster gradients
                in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
                )
                avg_incluster_gradients = average_of_weigts(copy.deepcopy(in_cluster_gradients))
                # Updating weights
                new_incluster_weights = aggrgator.optimize_weights(
                    weights=self.current_clusters[cluster_signature].model,
                    gradients = avg_incluster_gradients,
                    learning_rate = learning_rate,
                )
                # Update weights of the cluster model
                self.current_clusters[cluster_signature].model = copy.deepcopy(new_incluster_weights)
                # Updating weights for each node in the cluster
                for node_id, node in self.network.items():
                    if node.node_id in self.current_clusters[cluster_signature].nodes:
                        assert node_id == node.node_id
                        node.update_weights(new_incluster_weights)
                        self.node_cluster_mapping[iteration][node_id] = cluster_signature
            
            # # Step VI: Mixing the global model
            # avg_global_gradients = average_of_weigts(gradients)
            # # Updating the weights for the central model
            # new_global_weights = aggrgator.optimize_weights(
            #         weights=self.orchestrator_model.get_weights(),
            #         gradients = avg_global_gradients,
            #         learning_rate = learning_rate,
            #         )
            # self.orchestrator_model.update_weights(new_global_weights)
            
            # # Step VII: Post-training evaluation
            # # Preserving the orchestrator's model
            # self.orchestrator_model.store_model_on_disk(iteration=iteration,
            #                                             path=orchestrator_models_savepath)
            # Evaluating the new set of weights on local datasets.
            automatic_node_evaluation(
                iteration=iteration,
                nodes=self.network,
                save_path=os.path.join(metrics_savepath, "after_update_metrics.csv")
            )
            # # Evaluating the new set of weights on orchestrator's dataset.
            # evaluate_model(
            #     iteration=iteration,
            #     model=self.orchestrator_model,
            #     save_path=os.path.join(metrics_savepath, "orchestrator_metrics.csv"))
        
        # Step VIII: Out-of-loop (end of the training)
        with open(os.path.join(cluster_structure_savepath, "all_clustering_structures"), 'wb') as file:
            pickle.dump(self.all_clusters, file)
        save_nested_dict_ascsv(data=self.node_cluster_mapping, save_path=os.path.join(metrics_savepath, 'cluster_id_mapping.csv'))
        orchestrator_logger.critical("Training complete")
        return 0


    def training_protocol_briggs(
        self,
        iterations: int,
        sample_size: int,
        local_epochs: int,
        aggrgator: Aggregator,
        clustering_round: int,
        distance_threshold: float,
        learning_rate: float,
        metrics_savepath: str,
        nodes_models_savepath: str,
        orchestrator_models_savepath: str,
        sim_matrices_savepath: str,
        cluster_structure_savepath: str,
        batch_job = False
        ) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedOpt algorithm - it averages the local gradients 
        and aggregates them using a selecred optimizer.
        SOURCE: 

        Parameters
        ----------
        iterations: int
            Number of (global) iterations // epochs to train the models for.
        sample_size: int
            Size of the sample
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        aggregator: Aggregator
            Instance of the Aggregator object that will be used to aggregate the result each round
        clustering_round: int
            Round during which a clustering will occur
        learning_rate: float
            Learning rate to be used for optimization.
        metrics_savepath: str
            Path for saving the metrics
        nodes_models_savepath: str
            Path for saving the models in the .pt format.
        orchestrator_models_savepath: str
            Path for saving the orchestrator models.
        clustering_root_savepath: str
            Path for saving all the clustering results (general).
        sim_matrices_savepath: str
            Path for saving all the similarity matrices.
        cluster_structure_savepath: str
            Path for saving all the cluster structures.
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        # Step I: Create first cluster (the whole population)
        self.current_clusters = {'A': Cluster_Structure(
            population=np.array([node for node in self.network]), # First cluster, all nodes belonging to it.
            model = copy.deepcopy(self.orchestrator_model.get_weights()), # Model for the first cluster, direct copy of the central orchestrator model.
            iteration=0 # Created During the First Iteration
        )}
        self.all_clusters = {'A': self.current_clusters['A']} # List of all clusters that were created during the simulation
        self.node_cluster_mapping = OrderedDict() # mapping between iteration: {node: cluster_id}
        
        # Step II: Looping over the iterations
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")            
            # Sampling nodes
            sampled_nodes = sample_nodes(
                nodes = self.network,
                sample_size = sample_size,
                generator = self.generator
            )
            
            # Step III: Training nodes
            training_results, gradients = self.train_epoch(
                sampled_nodes=sampled_nodes,
                iteration=iteration,
                local_epochs=local_epochs,
                mode='gradients',
                save_model=True,
                save_path=nodes_models_savepath,
                batch_job=batch_job
            )
            # Preserving metrics of the training
            save_nested_dict_ascsv(
                data=training_results,
                save_path=os.path.join(metrics_savepath, 'training_metrics.csv')
            )
            # Testing nodes on the local dataset before the model update (only sampled nodes).
            automatic_node_evaluation(
                iteration=iteration,
                nodes=sampled_nodes,
                save_path=os.path.join(metrics_savepath, "before_update_metrics.csv")
            )
            # Evaluating the generalizability of the local model
            automatic_generalizability_evaluation(
                iteration=iteration,
                nodes=self.network,
                general_model=copy.deepcopy(self.orchestrator_model),
                save_path=os.path.join(metrics_savepath, "after_update_generalizability.csv")
            )
            
            if iteration == clustering_round:
                self.briggs_hierarchical(
                    iteration=iteration,
                    gradients=copy.deepcopy(gradients),
                    sim_mat_savepath=sim_matrices_savepath,
                    distance_threshold=distance_threshold
                )
            else:
                pass
            
            self.node_cluster_mapping[iteration] = {}
            loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iterating # Copy obj for iterating
            # Step IV: In-cluster aggregation
            for cluster_signature in loop_current_clusters_keys:
                # Collect in-cluster gradients
                in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
                )
                avg_incluster_gradients = average_of_weigts(copy.deepcopy(in_cluster_gradients))
                # Updating weights
                new_incluster_weights = aggrgator.optimize_weights(
                    weights=self.current_clusters[cluster_signature].model,
                    gradients = avg_incluster_gradients,
                    learning_rate = learning_rate,
                    )
                # Update weights of the cluster model
                self.current_clusters[cluster_signature].model = copy.deepcopy(new_incluster_weights)
                # Updating weights for each node in the cluster
                for node_id, node in self.network.items():
                    if node.node_id in self.current_clusters[cluster_signature].nodes:
                        assert node_id == node.node_id
                        node.update_weights(new_incluster_weights)
                        self.node_cluster_mapping[iteration][node_id] = cluster_signature
            
            # # Step VI: Mixing the global model
            # avg_global_gradients = average_of_weigts(gradients)
            # # Updating the weights for the central model
            # new_global_weights = aggrgator.optimize_weights(
            #         weights=self.orchestrator_model.get_weights(),
            #         gradients = avg_global_gradients,
            #         learning_rate = learning_rate,
            #         )
            # self.orchestrator_model.update_weights(new_global_weights)
            
            # # Step VII: Post-training evaluation
            # # Preserving the orchestrator's model
            # self.orchestrator_model.store_model_on_disk(iteration=iteration,
            #                                             path=orchestrator_models_savepath)
            # Evaluating the new set of weights on local datasets.
            automatic_node_evaluation(
                iteration=iteration,
                nodes=self.network,
                save_path=os.path.join(metrics_savepath, "after_update_metrics.csv")
            )
            # # Evaluating the new set of weights on orchestrator's dataset.
            # evaluate_model(
            #     iteration=iteration,
            #     model=self.orchestrator_model,
            #     save_path=os.path.join(metrics_savepath, "orchestrator_metrics.csv"))
        
        # Step VIII: Out-of-loop (end of the training)
        with open(os.path.join(cluster_structure_savepath, "all_clustering_structures"), 'wb') as file:
            pickle.dump(self.all_clusters, file)
        save_nested_dict_ascsv(data=self.node_cluster_mapping, save_path=os.path.join(metrics_savepath, 'cluster_id_mapping.csv'))
        orchestrator_logger.critical("Training complete")
        return 0


    def training_protocol_baseline(
        self,
        iterations: int,
        sample_size: int,
        local_epochs: int,
        aggrgator: Aggregator,
        learning_rate: float,
        metrics_savepath: str,
        nodes_models_savepath: str,
        orchestrator_models_savepath: str,
        sim_matrices_savepath: str,
        cluster_structure_savepath: str,
        skip_calculation: bool = True,
        batch_job = False
        ) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedOpt algorithm - it averages the local gradients 
        and aggregates them using a selecred optimizer.
        SOURCE: 

        Parameters
        ----------
        iterations: int
            Number of (global) iterations // epochs to train the models for.
        sample_size: int
            Size of the sample
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        aggregator: Aggregator
            Instance of the Aggregator object that will be used to aggregate the result each round
        learning_rate: float
            Learning rate to be used for optimization.
        metrics_savepath: str
            Path for saving the metrics
        nodes_models_savepath: str
            Path for saving the models in the .pt format.
        orchestrator_models_savepath: str
            Path for saving the orchestrator models.
        clustering_root_savepath: str
            Path for saving all the clustering results (general).
        sim_matrices_savepath: str
            Path for saving all the similarity matrices.
        cluster_structure_savepath: str
            Path for saving all the cluster structures.
        skip_calculation: bool = True
            If set to True, will skip calculation for the cosine distance matrix
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        # Step I: Create first cluster (the whole population)
        self.current_clusters = {'A': Cluster_Structure(
            population=np.array([node for node in self.network]), # First cluster, all nodes belonging to it.
            model = copy.deepcopy(self.orchestrator_model.get_weights()), # Model for the first cluster, direct copy of the central orchestrator model.
            iteration=0 # Created During the First Iteration
        )}
        self.all_clusters = {'A': self.current_clusters['A']} # List of all clusters that were created during the simulation
        self.node_cluster_mapping = OrderedDict() # mapping between iteration: {node: cluster_id}
        
        # Step II: Looping over the iterations
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")            
            # Sampling nodes
            sampled_nodes = sample_nodes(
                nodes = self.network,
                sample_size = sample_size,
                generator = self.generator
            )
            
            # Step III: Training nodes
            training_results, gradients = self.train_epoch(
                sampled_nodes=sampled_nodes,
                iteration=iteration,
                local_epochs=local_epochs,
                mode='gradients',
                save_model=True,
                save_path=nodes_models_savepath,
                batch_job=True
            )
            # Preserving metrics of the training
            save_nested_dict_ascsv(
                data=training_results,
                save_path=os.path.join(metrics_savepath, 'training_metrics.csv'))
            # Testing nodes on the local dataset before the model update (only sampled nodes).
            automatic_node_evaluation(
                iteration=iteration,
                nodes=sampled_nodes,
                save_path=os.path.join(metrics_savepath, "before_update_metrics.csv"))
            # Evaluating the generalizability of the local model
            automatic_generalizability_evaluation(
                iteration=iteration,
                nodes=self.network,
                general_model=copy.deepcopy(self.orchestrator_model),
                save_path=os.path.join(metrics_savepath, "after_update_generalizability.csv")
            )
            if skip_calculation:
                pass
            else:
                # Step IV: Clustering
                self.no_clustering(
                    iteratio = iteration,
                    gradients = copy.deepcopy(gradients),
                    sim_mat_savepath = sim_matrices_savepath,
                    temperature_save_path = metrics_savepath,
                )
            
            self.node_cluster_mapping[iteration] = {}
            loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iterating
            # Step V: In-cluster aggregation
            for cluster_signature in loop_current_clusters_keys:
                # Collect in-cluster gradients
                in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
                )
                avg_incluster_gradients = average_of_weigts(copy.deepcopy(in_cluster_gradients))
                # Updating weights
                new_incluster_weights = aggrgator.optimize_weights(
                    weights=self.current_clusters[cluster_signature].model,
                    gradients = avg_incluster_gradients,
                    learning_rate = learning_rate,
                    )
                # Update weights of the cluster model
                self.current_clusters[cluster_signature].model = copy.deepcopy(new_incluster_weights)
                # Updating weights for each node in the cluster
                for node_id, node in self.network.items():
                    if node.node_id in self.current_clusters[cluster_signature].nodes:
                        assert node_id == node.node_id
                        node.update_weights(new_incluster_weights)
                        self.node_cluster_mapping[iteration][node_id] = cluster_signature
            
            # # Step VI: Mixing the global model
            # avg_global_gradients = average_of_weigts(gradients)
            # # Updating the weights for the central model
            # new_global_weights = aggrgator.optimize_weights(
            #         weights=self.orchestrator_model.get_weights(),
            #         gradients = avg_global_gradients,
            #         learning_rate = learning_rate,
            #         )
            # self.orchestrator_model.update_weights(new_global_weights)
            
            # # Step VII: Post-training evaluation
            # # Preserving the orchestrator's model
            # self.orchestrator_model.store_model_on_disk(iteration=iteration,
            #                                             path=orchestrator_models_savepath)
            # Evaluating the new set of weights on local datasets.
            automatic_node_evaluation(
                iteration=iteration,
                nodes=self.network,
                save_path=os.path.join(metrics_savepath, "after_update_metrics.csv")
            )
            # # Evaluating the new set of weights on orchestrator's dataset.
            # evaluate_model(
            #     iteration=iteration,
            #     model=self.orchestrator_model,
            #     save_path=os.path.join(metrics_savepath, "orchestrator_metrics.csv"))
        
        # Step VIII: Out-of-loop (end of the training)
        with open(os.path.join(cluster_structure_savepath, "all_clustering_structures"), 'wb') as file:
            pickle.dump(self.all_clusters, file)
        save_nested_dict_ascsv(data=self.node_cluster_mapping, save_path=os.path.join(metrics_savepath, 'cluster_id_mapping.csv'))
        orchestrator_logger.critical("Training complete")
        return 0
    

    def training_protocol_sattler(
        self,
        iterations: int,
        sample_size: int,
        local_epochs: int,
        aggrgator: Aggregator,
        learning_rate: float,
        EPS1: float,
        EPS2: float,
        round_cooldown: int,
        metrics_savepath: str,
        nodes_models_savepath: str,
        orchestrator_models_savepath: str,
        sim_matrices_savepath: str,
        cluster_structure_savepath: str,
        batch_job = False
        ) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedOpt algorithm - it averages the local gradients 
        and aggregates them using a selecred optimizer.
        SOURCE: 

        Parameters
        ----------
        iterations: int
            Number of (global) iterations // epochs to train the models for.
        sample_size: int
            Size of the sample
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        aggregator: Aggregator
            Instance of the Aggregator object that will be used to aggregate the result each round
        learning_rate: float
            Learning rate to be used for optimization.
        metrics_savepath: str
            Path for saving the metrics
        nodes_models_savepath: str
            Path for saving the models in the .pt format.
        orchestrator_models_savepath: str
            Path for saving the orchestrator models.
        clustering_root_savepath: str
            Path for saving all the clustering results (general).
        sim_matrices_savepath: str
            Path for saving all the similarity matrices.
        cluster_structure_savepath: str
            Path for saving all the cluster structures.
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        # Step I: Create first cluster (the whole population)
        self.current_clusters = {'A': Cluster_Structure(
            population=np.array([node for node in self.network]), # First cluster, all nodes belonging to it.
            model = copy.deepcopy(self.orchestrator_model.get_weights()), # Model for the first cluster, direct copy of the central orchestrator model.
            iteration=0 # Created During the First Iteration
        )}
        self.all_clusters = {'A': self.current_clusters['A']} # List of all clusters that were created during the simulation
        self.node_cluster_mapping = OrderedDict() # mapping between iteration: {node: cluster_id}
        
        # Step II: Looping over the iterations
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")            
            # Sampling nodes
            sampled_nodes = sample_nodes(
                nodes = self.network,
                sample_size = sample_size,
                generator = self.generator
            )
            
            # Step III: Training nodes
            training_results, gradients = self.train_epoch(
                sampled_nodes=sampled_nodes,
                iteration=iteration,
                local_epochs=local_epochs,
                mode='gradients',
                save_model=True,
                save_path=nodes_models_savepath,
                batch_job=batch_job
            )
            # Preserving metrics of the training
            save_nested_dict_ascsv(
                data=training_results,
                save_path=os.path.join(metrics_savepath, 'training_metrics.csv'))
            # Testing nodes on the local dataset before the model update (only sampled nodes).
            automatic_node_evaluation(
                iteration=iteration,
                nodes=sampled_nodes,
                save_path=os.path.join(metrics_savepath, "before_update_metrics.csv"))
                        # Evaluating the generalizability of the local model
            automatic_generalizability_evaluation(
                iteration=iteration,
                nodes=self.network,
                general_model=copy.deepcopy(self.orchestrator_model),
                save_path=os.path.join(metrics_savepath, "after_update_generalizability.csv")
            )
            
            # Clustering
            self.sattler_clustering(
                iteration=iteration,
                gradients=copy.deepcopy(gradients),
                EPS_1=EPS1,
                EPS_2=EPS2,
                round_cooldown=round_cooldown,
                sim_mat_savepath=sim_matrices_savepath,
            )
                        
            self.node_cluster_mapping[iteration] = {}
            loop_current_clusters_keys = copy.deepcopy(list(self.current_clusters.keys())) # Copy obj for iterating # Copy obj for iterating
            # Step V: In-cluster aggregation
            for cluster_signature in loop_current_clusters_keys:
                # Collect in-cluster gradients
                in_cluster_gradients = select_gradients(
                nodes_id_list=self.current_clusters[cluster_signature].nodes,
                gradients=gradients
                )
                avg_incluster_gradients = average_of_weigts(copy.deepcopy(in_cluster_gradients))
                # Updating weights
                new_incluster_weights = aggrgator.optimize_weights(
                    weights=self.current_clusters[cluster_signature].model,
                    gradients = avg_incluster_gradients,
                    learning_rate = learning_rate,
                )
                # Update weights of the cluster model
                self.current_clusters[cluster_signature].model = copy.deepcopy(new_incluster_weights)
                # Updating weights for each node in the cluster
                for node_id, node in self.network.items():
                    if node.node_id in self.current_clusters[cluster_signature].nodes:
                        assert node_id == node.node_id
                        node.update_weights(new_incluster_weights)
                        self.node_cluster_mapping[iteration][node_id] = cluster_signature
            
            # # Step VI: Mixing the global model
            # avg_global_gradients = average_of_weigts(gradients)
            # # Updating the weights for the central model
            # new_global_weights = aggrgator.optimize_weights(
            #         weights=self.orchestrator_model.get_weights(),
            #         gradients = avg_global_gradients,
            #         learning_rate = learning_rate,
            #         )
            # self.orchestrator_model.update_weights(new_global_weights)
            
            # # Step VII: Post-training evaluation
            # # Preserving the orchestrator's model
            # self.orchestrator_model.store_model_on_disk(iteration=iteration,
            #                                             path=orchestrator_models_savepath)
            # Evaluating the new set of weights on local datasets.
            automatic_node_evaluation(
                iteration=iteration,
                nodes=self.network,
                save_path=os.path.join(metrics_savepath, "after_update_metrics.csv")
            )
            # Evaluating the new set of weights on orchestrator's dataset.
            # evaluate_model(
            #     iteration=iteration,
            #     model=self.orchestrator_model,
            #     save_path=os.path.join(metrics_savepath, "orchestrator_metrics.csv"))
        
        # Step VIII: Out-of-loop (end of the training)
        with open(os.path.join(cluster_structure_savepath, "all_clustering_structures"), 'wb') as file:
            pickle.dump(self.all_clusters, file)
        save_nested_dict_ascsv(data=self.node_cluster_mapping, save_path=os.path.join(metrics_savepath, 'cluster_id_mapping.csv'))
        orchestrator_logger.critical("Training complete")
        return 0