import numpy as np
from config import LINKS, CH_WIDTH_MHZ, PN_DBM, RSSI_THRESH_DBM, FAIR_STRATEGY
from actions import generate_action_set

ACTIONS = generate_action_set(LINKS)

def epsilon_for_iter(t):
    return 1.0 / np.sqrt(float(t))

def _random_nonempty_links(rng, num_links):
    while True:
        cand = rng.randint(0, 2, size=num_links)
        if cand.sum() != 0:
            return cand

def frl_select_actions(APs, t, rng, fair_strategy=FAIR_STRATEGY):
    eps = epsilon_for_iter(t)
    for ap in APs:
        if rng.rand() > eps:
            # exploitation
            if fair_strategy == 2:
                # avgFair -> choose MIN AccRew
                idx = int(np.argmin(ap.AccRew))
            else:
                # mini-max or propFair -> choose MAX AccRew
                idx = int(np.argmax(ap.AccRew))
            ap.links = ACTIONS[idx].copy()
            ap.stateFRL = idx + 1
            ap.frlVal = ap.AccRew[idx]
        else:
            # exploration
            ap.links = _random_nonempty_links(rng, LINKS)

def frl_step_bandwidth_and_rate(APs, STAs, NodeMatrix):
    """
    Same as rl_step_bandwidth_and_rate but also compute rateMax.
    """
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
        # theoretical best: full Links bandwidth, no interference
        ideal_denom = Pnlin
        ideal_term = APs[k].Prlin / ideal_denom
        APs[k].rateMax = (LINKS * CH_WIDTH_MHZ) * np.log2(1.0 + ideal_term)
    return rates

def _neighbors_for_ap(k, APs, NodeMatrix, rssi_thresh):
    """
    Return indices of APs in AP k's collaboration set (including itself),
    where NodeMatrix[k, i] >= RSSI_THRESH_DBM.
    We interpret NodeMatrix[k,i] == received power at AP k from AP i,
    exactly like your MATLAB getGlobalErrorNew.
    """
    nbrs = [k]
    for i in range(len(APs)):
        if i == k:
            continue
        if NodeMatrix[k, i] >= rssi_thresh:
            nbrs.append(i)
    return nbrs

def frl_compute_LLMs(APs, fair_strategy):
    """
    LLM per AP:
    - mini-max (1):  LLM = rate
    - avgFair (2):   LLM = (rateMax - rate)^2
    - propFair (3):  LLM = rate
    """
    for ap in APs:
        if fair_strategy == 1:          # mini-max
            ap.LLM = ap.rate
        elif fair_strategy == 2:        # avgFair (MSE to ideal)
            ap.LLM = (ap.rateMax - ap.rate)**2
        elif fair_strategy == 3:        # propFair (PF utility via log later)
            ap.LLM = ap.rate
        else:
            raise ValueError("Unknown fair_strategy")

def frl_compute_GLMs(APs, NodeMatrix, fair_strategy, rssi_thresh=RSSI_THRESH_DBM):
    """
    This is getGlobalErrorNew().
    For each AP k, we aggregate local metrics from neighbors in APCS.
    """
    N_APs = len(APs)
    for k in range(N_APs):
        nbrs = _neighbors_for_ap(k, APs, NodeMatrix, rssi_thresh)
        if fair_strategy == 1:
            # mini-max: choose min LLM (i.e. min rate) among neighbors
            vals = [APs[i].LLM for i in nbrs]
            APs[k].GLM = np.min(vals)
        elif fair_strategy == 2:
            # avgFair: mean of LLM (squared error)
            vals = [APs[i].LLM for i in nbrs]
            APs[k].GLM = np.mean(vals)
        elif fair_strategy == 3:
            # propFair: average log(rate) over neighbors
            logs = []
            for i in nbrs:
                # guard log(0) -> -inf blowup
                safe_rate = max(APs[i].LLM, 1e-12)
                logs.append(np.log(safe_rate))
            APs[k].GLM = np.mean(logs)
        else:
            raise ValueError("Unknown fair_strategy")

def frl_update_rewards(APs, t, fair_strategy):
    """
    Mirrors frlUpdateReward.m:
    For each AP, after GLM is computed:
      InsRew[t, s] = GLM
      ActnCount[s] += 1
      AccRew[s] = sum(InsRew[:,s]) / ActnCount[s]
    where s is the index of the chosen action.
    """
    for ap in APs:
        # which action was chosen?
        for s_idx, act in enumerate(ACTIONS):
            if np.array_equal(ap.links, act):
                ap.InsRew[t-1, s_idx] = ap.GLM
                ap.ActnCount[s_idx] += 1.0
                ap.AccRew[s_idx] = (
                    np.sum(ap.InsRew[:, s_idx]) /
                    ap.ActnCount[s_idx]
                )
                ap.frlVal = ap.AccRew[s_idx]
                break
