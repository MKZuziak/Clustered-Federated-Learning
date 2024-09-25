import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(
    data,
    title,
    iteration,
    path = None,
    ):
    clients = [node for node in range(30)]

    fig, ax = plt.subplots(figsize=(40, 40))
    im = ax.imshow(data)
    # Set the tickls
    ax.set_xticks(np.arange(len(clients)), labels=clients, fontsize=28)
    ax.set_yticks(np.arange(len(clients)), labels=clients, fontsize=28)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(clients)):
        for j in range(len(clients)):
            text = ax.text(j, i, round(data[i, j], 3),
                        ha="center", va="center", color="w")
    
    # Create a colorbar that will be placed next to a heatmap
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=40)
    # Add suptitle to the figure
    fig.suptitle(title, fontsize=70)
    ax.set_title(f"Iteration: {iteration}", fontsize=60)
    # Save the figure
    plt.savefig(path)
    # Close the figure to free-up memory
    plt.close()


def plot_clustered_matrices(
    cosine_data,
    cluster_data,
    title,
    iteration,
    path=None
    ):
    ids = np.array([node for node in range(30)])
    
    cluster_0_ids = ids[np.where(cluster_data == 0)]
    cluster_0_cos = cosine_data[np.where(cluster_data == 0)]
    cluster_1_ids = ids[np.where(cluster_data == 1)]
    cluster_1_cos = cosine_data[np.where(cluster_data == 1)]
    
    clustered_ids = np.hstack([cluster_0_ids, cluster_1_ids])
    clusetered_cos = np.vstack([cluster_0_cos, cluster_1_cos])
 
    fig, ax = plt.subplots(figsize=(40, 40))
    im = ax.imshow(clusetered_cos)
    # Set the tickls
    ax.set_xticks(np.arange(len(clustered_ids)), labels=clustered_ids, fontsize=28)
    ax.set_yticks(np.arange(len(clustered_ids)), labels=clustered_ids, fontsize=28)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(clustered_ids)):
        for j in range(len(clustered_ids)):
            text = ax.text(j, i, round(cosine_data[i, j], 3),
                        ha="center", va="center", color="w")
    
    # Create a colorbar that will be placed next to a heatmap
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=40)
    # Add suptitle to the figure
    fig.suptitle(title, fontsize=70)
    ax.set_title(f"Iteration: {iteration}", fontsize=60)
    # Save the figure
    plt.savefig(path)
    # Close the figure to free-up memory
    plt.close()    