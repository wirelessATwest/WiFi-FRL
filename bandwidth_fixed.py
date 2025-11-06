import numpy as np
from config import CH_WIDTH_MHZ, PN_DBM, LINKS

def step_fixed(APs, STAs, NodeMatrix):
    """
    Mutates APs in place to compute:
    - links = all ones
    - bw, interference, rate
    Returns array of rates [N_APs].
    """
    N_APs = len(APs)
    # noise in linear
    Pnlin = 10**(PN_DBM/10.0)

    # assign full bandwidth
    for k in range(N_APs):
        APs[k].links = np.ones(LINKS, dtype=int)
        APs[k].bw = int(np.sum(APs[k].links))
        # Pr taken same way as MATLAB: NodeMatrix(k, N_APs + k)
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
                Pilin = 10**(NodeMatrix[i + N_APs, j]/10.0)  # power at STA_i from AP_j
                APs[i].CCAinRange += Pilin * overlap

    # Shannon-like rate
    rates = np.zeros(N_APs, dtype=float)
    for k in range(N_APs):
        denom = Pnlin + (CH_WIDTH_MHZ * APs[k].CCAinRange)
        sinr_term = APs[k].Prlin / denom
        APs[k].rate = (APs[k].bw * CH_WIDTH_MHZ) * np.log2(1.0 + sinr_term)
        rates[k] = APs[k].rate
    return rates
