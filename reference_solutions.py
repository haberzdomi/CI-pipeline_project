from vvn_asdex import magfie
import math
from biotsavart_asdex import biotsavart_asdex
import os
def circular_current(R_max, nR, nphi, nZ, R_0, I_c, nseg):
    files_to_move = ["biotsavart.inp", "co_asd.dd", "cur_asd.dd"]
    #move original input files to temporary folder
    os.mkdir("C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\temporary")
    for filename in files_to_move:
        os.rename("C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\"+filename, "C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\temporary\\"+filename)
    
    #Create input files for test
    file1 = open("biotsavart.inp", "w")
    print(nR, nphi, nZ, file=file1)
    print(0, R_max, file=file1)
    print(-R_max, R_max, file=file1)
    file1.close()
    file2 = open("co_asd.dd", "w")
    print(nseg+1, file=file2)
    dphi = 2*math.pi/nseg
    for i in range(nseg):
        print(R_0*math.cos(i*dphi), R_0*math.sin(i*dphi), 0, 1, 1, file=file2)
    print(R_0, 0, 0, 0, 1, file=file2)  #connect last point to first point
    file2.close()
    file3 = open("cur_asd.dd", "w")
    print(I_c, file=file3)
    file3.close()


    biotsavart_asdex()


    #remove test input files
    for filename in files_to_move:
        os.remove(filename)

    #return original input files to their former place
    for filename in files_to_move:
        os.rename("C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\temporary\\"+filename, "C:\\Users\\franz\\OneDrive\\Uni\\Software Engineering in Physics\\worstpractice\\"+filename)
    os.rmdir("temporary")
    return

circular_current(4,2,2,32,4,7,64)