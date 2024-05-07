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
        subprocess.run(["python", "data_gen_teacher.py", "-n", str(num_agents), "-g", str(1)])

        # data_6013b64ce8, data_27dea4bcce, data_41f896f2be, data_9bd5f5ee5f, data_4af6f9d879, data_a71679fd65, data_46ef7fc2a7, data_935be5ac8f
        # data_eb9b7315c6, data_0b7d25ce95, data_89a277ffd9, data_cf84771ef1, data_76a571ea15, data_dbf79f7d01, data_5f734fb2a1, data_6644bb7ada