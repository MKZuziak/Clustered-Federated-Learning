from collections import OrderedDict

def select_gradients(
    nodes_id_list: list,
    gradients: OrderedDict,
    sort: bool = True
):
    in_cluster_gradients = OrderedDict()
    for node_id, gradient in gradients.items():
        if node_id in nodes_id_list:
            in_cluster_gradients[node_id] = gradient
    if sort == True:
        in_cluster_gradients = OrderedDict(sorted(in_cluster_gradients.items()))
    return in_cluster_gradients