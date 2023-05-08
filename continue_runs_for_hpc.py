import os
import shutil, errno
import time

num_runs = 50
simulation_name = "base_new"

# Make a new file for each simulation
def copy_file_or_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

for r_index in range(num_runs):
    driver_name = "simulations/" + simulation_name + "/driver" + str(r_index) + ".py"

    # Edit the driver file 
    a_file = open(driver_name, "r")
    list_of_lines = a_file.readlines()
    list_of_lines[0] = "import globalVars as globs\n"
    list_of_lines[1] = "import evolution_continue_simulation as ev\n"
    list_of_lines[2] = "import board as brd\n"
    list_of_lines[3] = "import hyperParameters as hp\n"

    a_file = open(driver_name, "w")
    a_file.writelines(list_of_lines)
    a_file.close()

    time.sleep(0.5)

    # Create a new submit file for each run
    submit_name = "simulations/" + simulation_name + "/submit" + str(r_index) + ".sh"

    time.sleep(0.5)

    # Run the submit file.
    run_submit_command = "sbatch " + submit_name
    os.system(run_submit_command)
