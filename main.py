from Mesh import Mesh
import numpy as np
from Eikonal import *
from PIL import Image


def main():
    #1.1
    #solve a maze with fast mraching
    #maze_solver_fmm(Image.open('maze.png'),[383,814],[233,8])
    #solve it with dijkastra
    #maze_solver_dij()


    #1.2
    #Optical_Path_Length()
    #1.3
    #easy_images = [np.array(Image.open('ball.png')), np.array(Image.open('CORONA.png'))]
    #images = [np.array(Image.open('duck.png')),np.array(Image.open('dog.png')) ]
    #for im in easy_images:
        #segmentation(im)


    meshes_names= ['tr_reg_000' , 'tr_reg_001']

    #1.4
    #save_geodesic(meshes_names)
    #embed_geodesic(meshes_names)
    cannonical_shape_MDS(meshes_names)


    #1.5
    # hands_up = 'tr_reg_079'
    #Farthest_Point_Sampling(hands_up,2000)



if __name__ == "__main__":
    main()