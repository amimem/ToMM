# data_gen_teacher.py
import argparse
# gen_datasets.py
import subprocess
import numpy as np


if __name__ == "__main__":
    for num_agents in np.linspace(1, 16, 16):
        print(f"generating data for {num_agents} agents")
        num_agents = int(num_agents)
        # for group in []:
        subprocess.run(["python", "data_gen_teacher.py", "-n", str(num_agents), "-g", str(num_agents)])