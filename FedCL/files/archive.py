import os
import time


def create_archive(
    path: str,
    archive_name: str = time.strftime("%d %m %Y %H %M %S", time.gmtime())
    ) -> tuple[str, str, str]:
    """Creates a basic directory structure for preserving simulation files.
    Returns Paths to metrics save location, nodes (models) save location and
    orchestrator (models) save location. The directory complies with the following
    structure:
    ---root
        ---archive_name
            ---results
            ---models
                ---orchestrator models
                ---nodes models
    
    Returns paths in order (metrics_savepath, nodes_models_savepath, 
    orchestrator_model_savepath).
    
    Parameters
    ----------
    path: str | Path
        Path-like object to the root directory in which the archive should be created
    archive_name: str (default to time.strftime("%d.%m.%Y %H:%M%S", time.gmtime())
        Name of the archive.
    
    Returns
    -------
    tuple[str, str str]
    """
    root = os.path.join(path, archive_name)
    assert not os.path.exists(root), f"Directory {root} already exists" 
    
    metrics_savepath = os.path.join(root, 'results')
    orchestrator_model_savepath = os.path.join(root, 'models', 'orchestrator')
    nodes_models_savepath = os.path.join(root, 'models', 'nodes')
    clustering_root = os.path.join(root, 'clustering')
    sim_matrices_savepath = os.path.join(clustering_root, 'similarity_matrices')
    cluster_structure_savepath = os.path.join(clustering_root, 'clustering_structure')
    
    os.makedirs(metrics_savepath)
    os.makedirs(nodes_models_savepath)
    os.makedirs(orchestrator_model_savepath)
    os.makedirs(clustering_root)
    os.makedirs(sim_matrices_savepath)
    os.makedirs(cluster_structure_savepath)
    
    return(metrics_savepath, 
           nodes_models_savepath, 
           orchestrator_model_savepath,
           clustering_root,
           sim_matrices_savepath,
           cluster_structure_savepath) 