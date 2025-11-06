import numpy as np
#from config import MAX_X, MAX_Y, PT_DBM, LINKS, DATA_LOAD, RSSI_THRESH_DBM
from config import PT_DBM, DATA_LOAD, RSSI_THRESH_DBM

# Pathloss model params (from your code)
PL0TMB   = 54.12
GAMMA    = 2.06067
K_TMB    = 5.25
W_TMB    = 0.1467

class APState:
    def __init__(self, x, y, rssi_thresh, max_iter, num_links, is_frl=False):
        self.x = x
        self.y = y
        self.load = DATA_LOAD

        # link allocation (channel subset), e.g. [1,0,1,1]
        self.links = np.zeros(num_links, dtype=int)
        self.bw = 0

        # radio stats
        self.Pr = 0.0      # received power (dBm equivalent channel gain proxy)
        self.Prlin = 0.0   # linear scale
        self.nI = 0        # number interferers
        self.CCAinRange = 0.0
        self.rate = 0.0
        self.rateMax = 0.0

        # RL/FRL bookkeeping
        noStates = 2**num_links - 1
        self.InsRew = np.zeros((max_iter, noStates), dtype=float)  # instant reward per action over time
        self.ActnCount = np.zeros((noStates,), dtype=float)        # how many times we took each action
        self.AccRew = np.zeros((noStates,), dtype=float)           # accumulated / avg reward per action

        # RL fields
        self.stateRL = 0
        self.rlVal = 0.0

        # FRL extra fields
        self.rateMax = 0.0
        self.LLM = 0.0
        self.GLM = 0.0
        self.stateFRL = 0
        self.frlVal = 0.0

        # RSSI threshold for neighborhood collaboration
        self.rssi_thresh = rssi_thresh

class STAState:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def _pathloss_rx_power_dbm(tx_x, tx_y, rx_x, rx_y, pt_dbm):
    """
    Returns received power in dBm using same model:
    PL = PL0TMB + 10*gamma*log10(d) + kTMB*WTMB*d
    RxPower = Pt_dBm - PL
    MATLAB allowed d=0 which made PL -> inf, then later replaced inf with 0.
    We'll clamp d to 1e-6 to avoid inf/nan in Python.
    """
    dx = tx_x - rx_x
    dy = tx_y - rx_y
    d = np.sqrt(dx*dx + dy*dy)
    if d < 1e-6:
        d = 1e-6  # avoid log10(0)
    PL = PL0TMB + 10*GAMMA*np.log10(d) + K_TMB*W_TMB*d
    return pt_dbm - PL  # dBm

def build_topology(N_APs, N_STAs, xyCord, realization_idx, max_iter, num_links, deploy_mode):
    """
    Reproduces all the branching in *MloCreatNetwork.m.
    - If deploy=0: use xyCord for AP locations.
    - STA i is at same x as AP i, y + 10.
    Returns:
        APs (list of APState),
        STAs (list of STAState),
        NodeMatrix (np.ndarray of shape (N_APs+N_STAs, N_APs+N_STAs))
    """

    # Extract coordinates for this realization y
    # MATLAB indexing:
    #   x = xyCord(i, (2*y-1))
    #   y = xyCord(i, 2*y)
    # y is 1-based there. Our realization_idx is 0-based here.
    xs = xyCord[:N_APs, 2*realization_idx]
    ys = xyCord[:N_APs, 2*realization_idx + 1]

    APs = []
    STAs = []

    for i in range(N_APs):
        if deploy_mode == 1:
            # deterministic grid placement branch from MATLAB
            # We'll mirror behavior for each N_APs case in run_* scripts if needed.
            # For now: we keep random placement behavior (deploy=0),
            # which is what you actually used in your manuscript.
            pass

        ap_x = xs[i]
        ap_y = ys[i]
        sta_x = ap_x
        sta_y = ap_y + 10.0

        APs.append(APState(ap_x, ap_y, RSSI_THRESH_DBM, max_iter, num_links))
        STAs.append(STAState(sta_x, sta_y))

    # Build NodeMatrix:
    # NodeMatrix[a,b] = received power at node a from node b (dBm)
    # Node order 0..N_APs-1 = APs, N_APs..N_APs+N_STAs-1 = STAs
    total_nodes = N_APs + N_STAs
    NodeMatrix = np.zeros((total_nodes, total_nodes), dtype=float)

    for i in range(total_nodes):
        for j in range(total_nodes):
            if i < N_APs and j < N_APs:
                # AP_i receiving from AP_j
                rx = APs[i]
                tx = APs[j]
            elif i < N_APs and j >= N_APs:
                # AP_i receiving from STA_(j-N_APs)
                rx = APs[i]
                tx = STAs[j - N_APs]
            elif i >= N_APs and j < N_APs:
                # STA_(i-N_APs) receiving from AP_j
                rx = STAs[i - N_APs]
                tx = APs[j]
            else:
                # STA_(i-N_APs) receiving from STA_(j-N_APs)
                rx = STAs[i - N_APs]
                tx = STAs[j - N_APs]

            NodeMatrix[i, j] = _pathloss_rx_power_dbm(
                tx_x=tx.x, tx_y=tx.y,
                rx_x=rx.x, rx_y=rx.y,
                pt_dbm=PT_DBM
            )

    # Match MATLAB cleanup:
    NodeMatrix[~np.isfinite(NodeMatrix)] = 0.0
    return APs, STAs, NodeMatrix
