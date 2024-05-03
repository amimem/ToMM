# data_gen_teacher.py
import argparse
# gen_datasets.py
import subprocess

if __name__ == "__main__":
    for num_agents in [4, 8, 16]:
        # for group in []:
        subprocess.run(["python", "data_gen_teacher.py", "-n", str(num_agents), "-g", str(num_agents)])