import torch
import numpy as np
import os
import argparse
import time
import itertools
import h5py
import hashlib
import yaml
import torch
from torch.distributions import Categorical
from collections import OrderedDict
from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy
from utils import numpy_scalar_to_python

parser = argparse.ArgumentParser(description="data generation parameters")
parser.add_argument("-c", "--coordination", type=str, default="", help="action coordination type")
parser.add_argument("-a", "--num_actions", type=int, default=8, help="number of actions")
parser.add_argument("-n","--num_teachers",type=int,default=1,help="number of teachers (ground agents)")
parser.add_argument( "-w", "--teacher_hidden_dim", type=int, default=16, help="number of teachers (ground agents)")
parser.add_argument( "-g", "--num_groups", type=int, default=1, help="number of coordinated teacher groups (abstract agents)")
parser.add_argument("-d", "--dim_state", type=int, default=16, help="state space dimension")
parser.add_argument("-s", "--num_states", type=int, default=2**16, help="number of states")
parser.add_argument("--num_seeds", type=int, default=5, help="number of seeds")
parser.add_argument("--seed", type=int, default=0xBAD5EED, help="seed")
parser.add_argument("--output", type=str, default="output/", help="output directory")
parser.add_argument("-t", "--tag", type=str, default="", help="tag to complement hash filename")
args = parser.parse_args()


def generate_teacher_data():
    # set variables (temporarily hard coded, for now, temporarily.)
    num_groups = args.num_groups
    num_teachers = args.num_teachers
    num_teacher_per_group = num_teachers // num_groups
    num_states = args.num_states
    dim_teacher_out = args.num_actions
    dim_teacher_hid = args.teacher_hidden_dim
    dim_teacher_inp = args.dim_state
    coordination = args.coordination

    # just something a little better than incrementing a seed for num_seeds
    # https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning
    seed_sequence = np.random.SeedSequence(args.seed).spawn(args.num_seeds)

    datasets = {}
    for seed_idx, seed_seq in enumerate(seed_sequence):
        print(f"running seed {seed_idx} of {len(seed_sequence)}")
        seed = int(seed_seq.generate_state(1)[0])
        # set rng for each seed
        rng = torch.Generator().manual_seed(seed)

        # generate state representations
        # each state is a normally distributed vector of size dim_teacher_inp
        states = torch.FloatTensor(num_states, dim_teacher_inp).normal_(generator=rng)

        # generate teacher policies
        # each teacher policy is a neural network with dim_teacher_inp input, dim_teacher_hid hidden, and dim_teacher_out output
        actions = []
        for _ in range(int(num_groups)):
            if coordination == "shared_input":
                shared_state_bias = torch.FloatTensor(1, dim_teacher_inp).normal_(
                    generator=rng
                )
            for _ in range(int(num_teacher_per_group)):
                model = torch.nn.Sequential(
                    torch.nn.Linear(dim_teacher_inp, dim_teacher_hid),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dim_teacher_hid, dim_teacher_out),
                    torch.nn.Softmax(dim=-1),
                )
                for n, p in model.named_parameters():
                    if "weight" in n:
                        torch.nn.init.normal_(p, generator=rng)
                    elif "bias" in n:
                        torch.nn.init.zeros_(p)

                with torch.inference_mode():
                    if coordination == "shared_input":
                        probs = model(states + shared_state_bias)
                    else:
                        probs = model(states)

                    actions.append(torch.argmax(probs, dim=-1))

        # save data as numpy arrays with the following shapes:
        # actions: (num_states, num_teachers)
        # states: (num_states, dim_teacher_inp)
        # timesteps: (num_states,)
        actions = torch.vstack(actions).T.numpy()
        timesteps = np.arange(num_states)
        states = states.numpy()
        datasets[f"dataset_{seed_idx}"] = {
            "seed": seed,
            "states": states,
            "actions": actions,
            "timesteps": timesteps,
        }

    return datasets


if __name__ == "__main__":

    output_path = os.path.join(os.getcwd(), args.output)
    os.makedirs(output_path, exist_ok=True)

    datasets = generate_teacher_data()

    # take all args except output path
    hash_dict = args.__dict__.copy()
    hash_dict.pop("output")

    # make dash_dict an ordered dict
    hash_dict = dict(sorted(hash_dict.items()))

    # get the hash of the hash_dict
    hash_var = hashlib.blake2s(str(hash_dict).encode(), digest_size=5).hexdigest()
    # get a timestamp - use this to [n] either make the output folder unique or [y] as file metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # combine the hash to get a unique filename
    tag = args.tag + "_" if args.tag else ""
    output_filename = f"data_{tag}{hash_var}"
    print("saving " + output_filename)

    output_dir = os.path.join(output_path, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # save the data
    filename = os.path.join(output_dir, "data" + ".h5")
    attrs_filename = os.path.join(output_dir, "config" + ".yaml")

    with h5py.File(filename, "w") as f, open(attrs_filename, "w") as yaml_file:
        f.attrs.update(args.__dict__)
        f.attrs["A"] = args.num_actions
        f.attrs["K"] = args.dim_state
        f.attrs["N"] = args.num_teachers

        f.attrs["timestamp"] = timestamp
        attrs_dict = {
            "file_attrs": {k: numpy_scalar_to_python(v) for k, v in f.attrs.items()}
        }
        attrs_dict["file_attrs"]["hash"] = hash_var
        for dataset_name, dataset in datasets.items():
            group = f.create_group(dataset_name)
            for key, value in dataset.items():
                group.create_dataset(key, data=value)
        yaml.dump(attrs_dict, yaml_file)
