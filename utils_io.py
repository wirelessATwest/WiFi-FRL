#import numpy as np

#def save_txt(fname, arr):
#    np.savetxt(fname, arr, fmt="%.6f")

import os
import numpy as np

def save_txt(output_dir, fname, arr):
    """
    output_dir: folder path where to put the file
    fname: just the filename like "accfixedMLOdataSetAPs8.txt"
    arr: numpy array to save
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, fname)
    np.savetxt(full_path, arr, fmt="%.6f")
