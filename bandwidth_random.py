import numpy as np
from config import CH_WIDTH_MHZ, PN_DBM, LINKS, SEED

_rng = np.random.RandomState(SEED)

def _random_nonempty_links(num_links):
    # replicate your while-loop randi([0 1],1,Links) until sum != 0
    while True:
        cand = _rng.randint(0, 2, size=num_links)
        if cand.sum() != 0:
            return cand

def step_random(APs, STAs, NodeMatrix):
    N_APs = len(APs)
    Pnlin = 10**(PN_DBM/10.0)

    # random links per AP
    for k in range(N_APs):
        APs[k].links = _random_nonempty_links(LINKS)
        APs[k].bw = int(np.sum(APs[k].links))
        APs[k].Pr = NodeMatrix[k, N_APs + k]
        APs[k].Prlin = 10**(APs[k].Pr/10.0)

    # interference
    for i in range(N_APs):
        APs[i].nI = 0
        APs[i].CCAinRange = 0.0
        for j in range(N_APs):
            if j == i:
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
