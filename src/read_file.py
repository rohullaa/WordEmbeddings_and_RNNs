with open("slurm-5940091.out") as file:
    data = file.readlines()
    
    f = open("res.txt", "w")
    for line in data:
        if line.startswith('2022-06-13'):
            f.write(line)
            
    f.close()

