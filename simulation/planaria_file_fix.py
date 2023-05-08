import os
PATH = "data/condensed_shapes/"

_, _, filenames = next(os.walk(PATH), (None, None, []))

for fn in filenames:
    with open(PATH + fn, 'rb+') as filehandle:
        if len(filehandle.readlines()) >= 5000:
            continue
    
        filehandle.seek(-2, os.SEEK_END)
        filehandle.truncate()

    with open(PATH + fn, "a") as file_object:
        file_object.write("]}")