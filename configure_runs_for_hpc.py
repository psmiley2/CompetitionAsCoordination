import os
import shutil, errno
import time

simulations = [
    {
        "name": "oct_10_control",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": 0,
        "useSignals": False,
    },
    {
        "name": "oct_10_signals",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_neighbor_noise_low",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": .02,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_neighbor_noise_med",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": .05,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_neighbor_noise_high",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": .1,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_reservoir_noise_low",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": .02,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_reservoir_noise_med",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": .05,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_reservoir_noise_high",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": .1,
        "genomeLookupNoise": 0,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_genome_noise_low",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": .02,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_genome_noise_med",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": .05,
        "useSignals": True,
    },
    {
        "name": "oct_10_signals_genome_noise_high",
        "runs": 10,
        "onlyInfinite": False,
        "onlyFinite": False,
        "useReservoirsAsInputs": False,
        "onlyUseSizeAsFitness": False,
        "targetSize": 400,
        "targetAspectRatio": 5,
        "useConsistency": True,
        "boardWidth": 100,
        "populationSize": 250, 
        "neighborAssessmentNoise": 0,
        "reservoirAssessmentNoise": 0,
        "genomeLookupNoise": .1,
        "useSignals": True,
    }
]

# Make a new file for each simulation
def copy_file_or_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

for s in simulations:
    # Create new simulation folder
    copy_file_or_dir("simulation", "simulations/" + s["name"])

    # Create the hyperparameter file  
    new_hp_name = "simulations/" + s["name"] + "/hyperParameters.py"
    copy_file_or_dir("hyperParameters.py", new_hp_name)

    # Configure the hyperparameters. 
    c_file = open(new_hp_name, "r")
    list_of_lines = c_file.readlines()
    for i, l in enumerate(list_of_lines):
        if l[:7] == "onlyInf":
            list_of_lines[i] = "onlyInfinite = "+ str(s["onlyInfinite"]) + " # hp in script\n"
        elif l[:7] == "onlyFin":
            list_of_lines[i] = "onlyFinite = " + str(s["onlyFinite"]) + " # hp in script\n"
        elif l[:7] == "useRese":
            list_of_lines[i] = "useReservoirsAsInputs = " + str(s["useReservoirsAsInputs"]) + " # hp in script\n"
        elif l[:7] == "onlyUse":
            list_of_lines[i] = "onlyUseSizeAsFitness = " + str(s["onlyUseSizeAsFitness"]) + " # hp in script\n"
        elif l[:10] == "targetSize":
            list_of_lines[i] = "targetSize = " + str(s["targetSize"]) + " # hp in script\n"
        elif l[:10] == "targetAspe":
            list_of_lines[i] = "targetAspectRatio = " + str(s["targetAspectRatio"]) + " # hp in script\n"
        elif l[:10] == "useConsist":
            list_of_lines[i] = "useConsistency = " + str(s["useConsistency"]) + " # hp in script\n"
        elif l[:10] == "boardWidth":
            list_of_lines[i] = "boardWidth = " + str(s["boardWidth"]) + " # hp in script\n"
        elif l[:10] == "population":
            list_of_lines[i] = "populationSize = " + str(s["populationSize"]) + " # hp in script\n"
        elif l[:23] == "neighborAssessmentNoise":
            list_of_lines[i] = "neighborAssessmentNoise = " + str(s["neighborAssessmentNoise"]) + " # hp in script\n"
        elif l[:24] == "reservoirAssessmentNoise":
            list_of_lines[i] = "reservoirAssessmentNoise = " + str(s["reservoirAssessmentNoise"]) + " # hp in script\n"
        elif l[:17] == "genomeLookupNoise":
            list_of_lines[i] = "genomeLookupNoise = " + str(s["genomeLookupNoise"]) + " # hp in script\n"
        elif l[:10] == "useSignals":
            list_of_lines[i] = "useSignals = " + str(s["useSignals"]) + " # hp in script\n"
            

    c_file = open(new_hp_name, "w")
    c_file.writelines(list_of_lines)
    c_file.close()

    time.sleep(0.5)

    for r_index in range(s["runs"]):
        # Create a new driver for each run
        new_driver_name = "simulations/" + s["name"] + "/driver" + str(r_index) + ".py"
        copy_file_or_dir("driver.py", new_driver_name)

        # Edit the driver file 
        a_file = open(new_driver_name, "r")
        list_of_lines = a_file.readlines()
        for i, l in enumerate(list_of_lines):
            if l[:3] == "dst":
                list_of_lines[i] = "dst = \"/cluster/home/psmile01/simulations/" + s["name"] + "/data/r" + str(r_index) + ".json\""
        a_file = open(new_driver_name, "w")
        a_file.writelines(list_of_lines)
        a_file.close()

        # Create a new submit file for each run
        new_submit_name = "simulations/" + s["name"] + "/submit" + str(r_index) + ".sh"
        copy_file_or_dir("submit.sh", new_submit_name)

        time.sleep(0.5)

        # Edit the submit file 
        b_file = open(new_submit_name, "r")
        list_of_lines = b_file.readlines()
        for i, l in enumerate(list_of_lines):
            if l[:3] == "pyt":
                list_of_lines[i] = "python3 /cluster/home/psmile01/" + new_driver_name
        b_file = open(new_submit_name, "w")
        b_file.writelines(list_of_lines)
        b_file.close()


        # Run the submit file.
        run_submit_command = "sbatch " + new_submit_name
        os.system(run_submit_command)


