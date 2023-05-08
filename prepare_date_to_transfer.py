import os
import shutil

location = "simulations"

for sim in os.listdir(location):
    for f in os.listdir(location + "/" + sim):
        if f != "data":
            try:
                os.remove(location + "/" + sim + "/" + f)
                print("deleting file: ", f)
            except:
                shutil.rmtree(location + "/" + sim + "/" + f)
                print("deleting directory: ", f)

    for r in os.listdir(location + "/" + sim + "/data"):
        shutil.move(location + "/" + sim + "/data/" + r, location + "/" + sim)

    os.rmdir(location + "/" + sim + "/data")


