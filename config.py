import numpy as np

# Transmit / noise power (dBm)
PT_DBM = 20        # Pt
PN_DBM = -96       # Pn

# Geometry
MAX_X = 200
MAX_Y = 200

# Simulation control
MAX_SIM = 500      # number of realizations (y loop)
MAX_ITER = 2000    # time steps per realization
SEED = 100         # fixed RNG seed used in all your MATLAB scripts

# Network sizes to sweep
#AP_COUNTS = [2, 4, 8, 12, 16]  # mN_APs
#STA_COUNTS = [2, 4, 8, 12, 16] # nN_STAs (1 STA per AP effectively)
AP_COUNTS = [4]  # mN_APs
STA_COUNTS = [4] # nN_STAs (1 STA per AP effectively)

# Traffic / PHY
DATA_LOAD = 12000        # packet size / offered load per AP (not explicitly used in rate calc, but we keep it)
LINKS = 2                # number of 20 MHz chunks available
CH_WIDTH_MHZ = 80        # MHz per chunk
RSSI_THRESH_DBM = -82    # collaboration / interference range threshold for FRL (your code uses -82)
DEPLOY_MODE = 0          # 0 = random coords (xyCord), 1 = fixed grid

# RL/FRL settings
GREEDY_MODE = 1          # 1 = epsilon-greedy
FRL_ENABLED = 1
FAIR_STRATEGY = 1        # 1 = mini-max, 2 = avgFair, 3 = propFair

# Reproducible coordinate generation
def generate_xyCord(max_sim=MAX_SIM, seed=SEED):
    """
    Recreates what xyCoordinates16.mat did for you:
    Returns xyCord of shape (16, 2*max_sim),
    where row i is AP i, and for realization y:
      x = xyCord[i, 2*y]
      y = xyCord[i, 2*y+1]
    We generate uniform random positions in [0, MAX_X]x[0, MAX_Y],
    same spirit as deploy=0 in your MATLAB code.
    """
    rng = np.random.RandomState(seed)
    xyCord = np.zeros((16, 2*max_sim), dtype=float)
    for y in range(max_sim):
        # place 16 APs uniformly
        xs = rng.uniform(0, MAX_X, size=16)
        ys = rng.uniform(0, MAX_Y, size=16)
        xyCord[:, 2*y]   = xs
        xyCord[:, 2*y+1] = ys
    return xyCord
