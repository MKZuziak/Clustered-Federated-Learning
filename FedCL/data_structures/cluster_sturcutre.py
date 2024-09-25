from typing import Any

class Cluster_Structure():
    def __init__(
        self,
        population: list,
        model: Any,
        iteration: int,
        parent: Any = None
        ) -> None:
        self.nodes = population
        self.model = model
        self.clustering_iteration = iteration
        self.parent = parent
        self.children = None