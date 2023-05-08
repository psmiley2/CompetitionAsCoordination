import json 

FP = "data/base_new/r1.json"


for s_idx in range(50):
    FP = "data/base_new/r" + str(s_idx) + ".json"
    
    lines_to_delete = []
    with open(FP, "r") as file:
        data = json.load(file)["genomes"]

        for i in range(len(data) - 1):
            if data[i]["generation"] == data[i + 1]["generation"]:
                lines_to_delete.append(i)

    with open(FP, "r") as f:
        lines = f.readlines()

    print(lines_to_delete)
    print(list(reversed(lines_to_delete)))

    with open(FP, "w") as f:
        for ltd in list(reversed(lines_to_delete)):

            del lines[ltd + 1]

        for line in lines:
            f.write(line)

    lines_to_delete = []

    with open(FP, "r") as file:
        data = json.load(file)["genomes"]

        for i in range(len(data) - 1):
            if data[i]["generation"] == data[i + 1]["generation"]:
                lines_to_delete.append(i)

    print(lines_to_delete)
        