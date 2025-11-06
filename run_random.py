import numpy as np
from config import (
    AP_COUNTS, STA_COUNTS,
    MAX_SIM, MAX_ITER, SEED,
    LINKS
)
from config import generate_xyCord
from environment import build_topology
from bandwidth_random import step_random
from utils_io import save_txt

def run_random(output_dir):
    xyCord = generate_xyCord(MAX_SIM, SEED)

    for idx, N_APs in enumerate(AP_COUNTS):
        N_STAs = STA_COUNTS[idx]

        randMLOdata = np.zeros((MAX_ITER, N_APs), dtype=float)
        accRandMLOdata = np.zeros((MAX_SIM, N_APs), dtype=float)
        cdfRandMLOdata = np.zeros((MAX_ITER, MAX_SIM * N_APs), dtype=float)

        for y in range(MAX_SIM):
            APs, STAs, NodeMatrix = build_topology(
                N_APs=N_APs,
                N_STAs=N_STAs,
                xyCord=xyCord,
                realization_idx=y,
                max_iter=MAX_ITER,
                num_links=LINKS,
                deploy_mode=0
            )

            for t in range(MAX_ITER):
                rates = step_random(APs, STAs, NodeMatrix)
                randMLOdata[t, :] = rates
                cdfRandMLOdata[t, y*N_APs:(y+1)*N_APs] = rates

            accRandMLOdata[y, :] = np.mean(randMLOdata, axis=0)

        save_txt(output_dir, f"accRandMLOdataSetAPs{N_APs}.txt", accRandMLOdata)
        save_txt(output_dir, f"cdfRandMLOdataSetAPs{N_APs}.txt", cdfRandMLOdata)

if __name__ == "__main__":
    run_random(output_dir="logs/single_run_manual/random")
