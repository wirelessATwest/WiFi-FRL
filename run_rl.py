import numpy as np
from config import (
    AP_COUNTS, STA_COUNTS,
    MAX_SIM, MAX_ITER, SEED,
    LINKS
)
from config import generate_xyCord
from environment import build_topology
from agent_bandwidth_rl import (
    rl_select_actions,
    rl_step_bandwidth_and_rate,
    rl_update_rewards,
)
from utils_io import save_txt

def run_rl(output_dir):
    rng = np.random.RandomState(SEED)
    xyCord = generate_xyCord(MAX_SIM, SEED)

    for idx, N_APs in enumerate(AP_COUNTS):
        N_STAs = STA_COUNTS[idx]

        rlMLOdata = np.zeros((MAX_ITER, N_APs), dtype=float)
        accRlMLOdata = np.zeros((MAX_SIM, N_APs), dtype=float)
        cdfRlMLOdata = np.zeros((MAX_ITER, MAX_SIM * N_APs), dtype=float)

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

            for t in range(1, MAX_ITER+1):
                rl_select_actions(APs, t, rng)
                rates = rl_step_bandwidth_and_rate(APs, STAs, NodeMatrix)
                rl_update_rewards(APs, t)

                rlMLOdata[t-1, :] = rates
                cdfRlMLOdata[t-1, y*N_APs:(y+1)*N_APs] = rates

            accRlMLOdata[y, :] = np.mean(rlMLOdata, axis=0)

        save_txt(output_dir, f"accRlMLOdataSetAPs{N_APs}.txt", accRlMLOdata)
        save_txt(output_dir, f"cdfRlMLOdataSetAPs{N_APs}.txt", cdfRlMLOdata)

if __name__ == "__main__":
    run_rl(output_dir="logs/single_run_manual/rl")
