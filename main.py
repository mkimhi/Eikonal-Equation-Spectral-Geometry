from Mesh import Mesh
import numpy as np
from  Eikonal import *

def main():
    #solve a maze with fast mraching
    #tau_fm= fast_marching(show==True)
    maze_solver_fmm('maze.png',(383,814),(233,8))
    #solve it with dijkastra
    #maze_solver_dij()
    #Optical_Path_Length()

    #im = np.array(Image.open('CORONA.png'))
    #segmentation(im)

if __name__ == "__main__":
    main()