files_to_continue = []

num_files = 0

for i in range(100):
    try:
        file = open("data/inf_only_cons/r"+ str(i) + ".json","r")
        Counter = 0
        
        # Reading from file
        Content = file.read()
        CoList = Content.split("\n")
        
        for j in CoList:
            if j:
                Counter += 1
        
        if Counter < 7503:
            files_to_continue.append("r" + str(i))
                
        print("This is the number of lines in the file", Counter)
        num_files += 1
    except:
        pass

print(files_to_continue)
print("num files: ", num_files)