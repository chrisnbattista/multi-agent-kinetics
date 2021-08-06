import numpy as np
import torch
import os, glob
from typing import Dict, Any
import hashlib
import json
from . import worlds, forces

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary.
    Source: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    """
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def save_world(world, root_path, params, seed):
    '''
    Saves the specified world object in a specially-marked subdirectory of root_path.
    '''

    save_dir = f"{root_path}/{dict_hash(params)}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    np.savetxt(f"{save_dir}/{seed}.csv",
        world.get_full_history_with_indicators(),
        comments='',
        delimiter=',',
        fmt='%10.6f',
        header=','.join(list(world.schema) + list(world.indicator_schema))
    )

    with open(f"{save_dir}/params.json", 'w') as outfile:
        json.dump(params, outfile)

def load_world(filepath):
    '''
    ONLY WORKS ON 2D RN
    TO DO: FIX FORCES
    Creates a World object based on a specified file and the params.json in its containing folder.
    '''

    # Get sim params
    with open(os.path.dirname(filepath)+'/params.json', 'r') as infile:
        params = json.load(infile)
    
    # Get sim data
    data = torch.tensor(
        np.loadtxt(
            filepath,
            delimiter=',',
            skiprows=1
        )
    )
    history = data[:,:7]
    try:
        indicator_history = data[::len(torch.unique(history[:,1])),7:]
    except:
        indicator_history = torch.empty( (params['n_timesteps'], 0) )

    # Reconstruct simulation
    world = worlds.World(
        n_agents=len(torch.unique(data[:,1])),
        n_timesteps=params['n_timesteps'],
        timestep=params['timestep'],
        forces=[lambda world: forces.pairwise_world_lennard_jones_force(world, epsilon=params['epsilon'], sigma=params['sigma'])]
    )
    world.history = history
    world.indicator_history = indicator_history
    world.indicators = data[0,7:]
    world.current_timestep = params['n_timesteps'] - 1 

    return world, params

def list_worlds(dirpath):
    '''
    Recursively return all saved worlds in a folder.
    '''
    print(os.path.join(dirpath, "**/*.csv"))
    return glob.glob(os.path.join(dirpath, "**/*.csv"), recursive=True)