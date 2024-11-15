# Writing Your Own Utility Functions
Here, we present a few practical tips that may come in handy when projecting your own simulation

### Writing your own dataset loading method

To change the dataset loading method you should change the Federated Model attach_dataset_id method. This method always allows to assign a particular ID and a dataset to a node, hence it is important to remember, that Forcha Base assigns those two values at one method call (otherwise, you may mistakanely assign a certain dataset to a wrong client, which may influence the simulation's outcome). The baseline attach_dataset_id method is as follows:

```
    def attach_dataset_id(
        self,
        local_dataset: Any,
        node_name: int | str,
        only_test: bool = False,
        batch_size: int = 32
        ) -> None:

		### Here you should input your code
		self.trainloader = torch.utils.data.DataLoader(
                	your_training_data,
                	batch_size=batch_size,
                	shuffle=True,
                	num_workers=0,
            	)
		self.testloader = torch.utils.data.DataLoader(
                	your_test_data,
                	batch_size=batch_size,
                	shuffle=False,
                	num_workers=0,
            	)
```

Irrespective of how the data is passed to the method call, it should always be loaded using torch.utils.data.DataLoader. Otherwise, you may need to write a fully custom training loop.

### Metrics and data structures guidelines

The use of internal data structures should be (if possible) coherent across the whole library. Here, we provide a few baseline examples.

```
SimulationInstance.network <- dict[id: FederatedNode]
```

SimulationInstance.network is a dictionary mapping each node to its respective identification. It is also a central attribute of the Simulation instance, as it preserves all the nodes participating in the training over the whole instance's lifecycle.

`weights <- dict[id: OrderedDict]`

Weights (or gradients) of the participant should always be passed in the form of a Python dictionary, mapping the node's ID number to an OrderedDict containing the particular set of weights. This guarantees integrity of all the functions that either accept weights as their argument or return after processing.

```
training_results <- [node_id] = 
			{
			"iteration": [int] iteration,
			"node_id": [int] node_id
			"loss": [float] loss_list[-1],  
			"accuracy": [float] accuracy_list[-1],  
			"full_loss": [list] loss_list,  
			"full_accuracy": [list] accuracy_list}
```
