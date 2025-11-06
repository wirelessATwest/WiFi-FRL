import numpy as np
from config import LINKS, CH_WIDTH_MHZ, PN_DBM
from actions import generate_action_set

ACTIONS = generate_action_set(LINKS)  # shape (2^Links-1, Links)

def epsilon_for_iter(t):
    # epsilon = 1/sqrt(i)
    return 1.0 / np.sqrt(float(t))

def _random_nonempty_links(rng, num_links):
    while True:
        cand = rng.randint(0, 2, size=num_links)
        if cand.sum() != 0:
            return cand

def rl_select_actions(APs, t, rng):
    eps = epsilon_for_iter(t)
    for ap in APs:
        if rng.rand() > eps:
            # exploit: pick argmax AccRew
            # ties go to first max, which matches MATLAB behavior
            idx = int(np.argmax(ap.AccRew))
            ap.links = ACTIONS[idx].copy()
            ap.stateRL = idx + 1  # MATLAB is 1-based
            ap.rlVal = ap.AccRew[idx]
        else:
            # explore: random non-empty
            ap.links = _random_nonempty_links(rng, LINKS)
            # stateRL / rlVal not strictly needed during exploration

def rl_step_bandwidth_and_rate(APs, STAs, NodeMatrix):
    N_APs = len(APs)
    Pnlin = 10**(PN_DBM/10.0)

    # compute bw and received power
    for i in range(N_APs):
        APs[i].bw = int(np.sum(APs[i].links))
        APs[i].Pr = NodeMatrix[i, N_APs + i]
        APs[i].Prlin = 10**(APs[i].Pr/10.0)

    # interference
    for i in range(N_APs):
        APs[i].nI = 0
        APs[i].CCAinRange = 0.0
        for j in range(N_APs):
            if i == j:
                continue
            overlap = np.sum(APs[i].links * APs[j].links)
            if overlap != 0:
                APs[i].nI += 1
                Pilin = 10**(NodeMatrix[i + N_APs, j]/10.0)
                APs[i].CCAinRange += Pilin * overlap

    rates = np.zeros(N_APs, dtype=float)
    for k in range(N_APs):
        denom = Pnlin + (CH_WIDTH_MHZ * APs[k].CCAinRange)
        sinr_term = APs[k].Prlin / denom
        APs[k].rate = (APs[k].bw * CH_WIDTH_MHZ) * np.log2(1.0 + sinr_term)
        rates[k] = APs[k].rate
    return rates

def rl_update_rewards(APs, t):
    # update InsRew, ActnCount, AccRew
    # match rlUpdateReward.m
    for ap in APs:
        # find which action in ACTIONS matches ap.links
        # we'll do exact row compare
        for s_idx, act in enumerate(ACTIONS):
            if np.array_equal(ap.links, act):
                ap.InsRew[t-1, s_idx] = ap.rate
                ap.ActnCount[s_idx] += 1.0
                # accumulated average reward
                ap.AccRew[s_idx] = (
                    np.sum(ap.InsRew[:, s_idx]) /
                    ap.ActnCount[s_idx]
                )
                ap.rlVal = ap.AccRew[s_idx]
                break
