import numpy as np
from config import (
    AP_COUNTS, STA_COUNTS,
    MAX_SIM, MAX_ITER, SEED,
    LINKS, FAIR_STRATEGY
)
from config import generate_xyCord
from environment import build_topology
from agent_bandwidth_frl import (
    frl_select_actions,
    frl_step_bandwidth_and_rate,
    frl_compute_LLMs,
    frl_compute_GLMs,
    frl_update_rewards,
)
from utils_io import save_txt

def run_frl(output_dir):
    rng = np.random.RandomState(SEED)
    xyCord = generate_xyCord(MAX_SIM, SEED)

    for idx, N_APs in enumerate(AP_COUNTS):
        N_STAs = STA_COUNTS[idx]

        frlMLOdata = np.zeros((MAX_ITER, N_APs), dtype=float)
        accFrlMLOdata = np.zeros((MAX_SIM, N_APs), dtype=float)
        cdfFrlMLOdata = np.zeros((MAX_ITER, MAX_SIM * N_APs), dtype=float)

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
                frl_select_actions(APs, t, rng, fair_strategy=FAIR_STRATEGY)
                rates = frl_step_bandwidth_and_rate(APs, STAs, NodeMatrix)
                frl_compute_LLMs(APs, FAIR_STRATEGY)
                frl_compute_GLMs(APs, NodeMatrix, FAIR_STRATEGY)
                frl_update_rewards(APs, t, FAIR_STRATEGY)

                frlMLOdata[t-1, :] = rates
                cdfFrlMLOdata[t-1, y*N_APs:(y+1)*N_APs] = rates

            accFrlMLOdata[y, :] = np.mean(frlMLOdata, axis=0)

        # choose filenames based on fairness strategy like in your MATLAB
        if FAIR_STRATEGY == 1:
            prefix_acc = "accMiniMaxFrlMLOdataSetAPs"
            prefix_cdf = "cdfMiniMaxFrlMLOdataSetAPs"
        elif FAIR_STRATEGY == 2:
            prefix_acc = "accAvgFairFrlMLOdataSetAPs"
            prefix_cdf = "cdfAvgFairFrlMLOdataSetAPs"
        else:
            prefix_acc = "accPropFairFrlMLOdataSetAPs"
            prefix_cdf = "cdfPropFairFrlMLOdataSetAPs"

        save_txt(output_dir, f"{prefix_acc}{N_APs}.txt", accFrlMLOdata)
        save_txt(output_dir, f"{prefix_cdf}{N_APs}.txt", cdfFrlMLOdata)

if __name__ == "__main__":
    run_frl(output_dir="logs/single_run_manual/frl")
