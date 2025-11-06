import os
from datetime import datetime


from run_fixed import run_fixed
from run_random import run_random
from run_rl import run_rl
from run_frl import run_frl

def main():
    print("Starting Simulations....")
    
    # 1. make timestamped root log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_root, exist_ok=True)
    
    # 2. create scheme subfolders
    fixed_dir = os.path.join(log_root, "fixed")
    random_dir = os.path.join(log_root, "random")
    rl_dir = os.path.join(log_root, "rl")
    frl_dir = os.path.join(log_root, "frl")

    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)
    os.makedirs(rl_dir, exist_ok=True)
    os.makedirs(frl_dir, exist_ok=True)

        
    print("Simulating Fixed LA Scheme")
    run_fixed(output_dir=fixed_dir)
    print(">>> Completed <<<")

    #print("Simulating Random LA Scheme")
    #run_random(output_dir=random_dir)
    #print(">>> Completed <<<")

    #print("Simulating RL-based LA Scheme")
    #run_rl(output_dir=rl_dir)
    #print(">>> Completed <<<")

    #print("Simulating FRL-based LA Scheme")
    #run_frl(output_dir=frl_dir)
    #print(">>> Completed <<<")
    
    #print("All results saved under:", log_root)


if __name__ == "__main__":
    main()
