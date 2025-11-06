import numpy as np
from config import (
    AP_COUNTS, STA_COUNTS,
    MAX_SIM, MAX_ITER, SEED,
    LINKS
    #, CH_WIDTH_MHZ
)
from config import generate_xyCord
from environment import build_topology
from bandwidth_fixed import step_fixed
from utils_io import save_txt

def run_fixed(output_dir):
    #rng = np.random.RandomState(SEED)
    xyCord = generate_xyCord(MAX_SIM, SEED)

    for idx, N_APs in enumerate(AP_COUNTS):
        N_STAs = STA_COUNTS[idx]

        fixedMLOdata = np.zeros((MAX_ITER, N_APs), dtype=float)
        accFixedMLOdata = np.zeros((MAX_SIM, N_APs), dtype=float)
        cdfFixedMLOdata = np.zeros((MAX_ITER, MAX_SIM * N_APs), dtype=float)

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
                rates = step_fixed(APs, STAs, NodeMatrix)
                fixedMLOdata[t, :] = rates
                cdfFixedMLOdata[t, y*N_APs:(y+1)*N_APs] = rates

            accFixedMLOdata[y, :] = np.mean(fixedMLOdata, axis=0)

        # save inside this run's /fixed/ folder
        save_txt(output_dir, f"accfixedMLOdataSetAPs{N_APs}.txt", accFixedMLOdata)
        save_txt(output_dir, f"cdfFixedMLOdataSetAPs{N_APs}.txt", cdfFixedMLOdata)

if __name__ == "__main__":
    # standalone usage fallback:
    run_fixed(output_dir="logs/single_run_manual/fixed")