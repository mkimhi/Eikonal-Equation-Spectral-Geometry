import numpy as np
import matplotlib.pyplot as plt
from Mesh import Mesh
import eikonalfm
import cv2
import networkx as nx
from PIL import Image

"""
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import eye as sparse_eye
from matplotlib import gridspec
import matplotlib
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
"""

def fast_marching():
    wall_weight = 1000
    path_weight = 1
    source_point = np.array([383,814])
    target_point = np.array([233, 8])
    dx = (1.0, 1.0)
    order = 2
    maze = np.asanyarray(Image.open('maze.png').convert('1')).astype('double')
    maze[maze == 1] = wall_weight
    maze[maze == 0] = path_weight
    tau_fm_maze = eikonalfm.fast_marching(maze, source_point, dx, order)
    tau0_maze = eikonalfm.distance(tau_fm_maze.shape, dx, source_point, indexing="ij")
    # plt.imshow(tau0_maze * tau_fm_maze,origin ="lower")

    plt.imshow(tau_fm_maze,cmap="jet")
    plt.colorbar()
    plt.show()
    return tau_fm_maze


def maze_solver_fmm(tau_fm):
    im=np.array(Image.open('maze.png'))
    c = np.asanyarray(Image.open('maze.png').convert('1')).astype('double')
    image =np.zeros(c.shape)

    grad_x, grad_y = np.gradient(tau_fm)
    location = (233,8)
    for _ in range(100000):
        image[location[0], location[1]] = 1
        grad = np.array([grad_x[location[0], location[1]], grad_y[location[0], location[1]]])
        grad_abs = np.abs(grad)
        grad_abs[grad_abs == 0] = 1 #not devide by 0
        location -= (grad / grad_abs).astype(int)
    im[image==1] = 255,0,0
    plt.imshow(im)
    plt.show()

def maze_solver_dij():
    c = np.asanyarray(Image.open('maze.png').convert('1')).astype('double')
    G = nx.Graph()
    indexes = np.array(np.where(c==1))
    nodes = [tuple(indexes[:,i]) for i in range(len(indexes[0]))]
    G.add_nodes_from(nodes)

    hight, width = c.shape
    for i in range(hight-1):
        for j in range(width-1):
            if c[i,j] == 1:
                if c[i+1,j]==1:
                    G.add_edge((i,j),(i+1,j))
                if c[i,j+1]==1:
                    G.add_edge((i, j), (i, j+1))
    image = np.zeros(c.shape)
    im=np.array(Image.open('maze.png'))
    for idx in nx.dijkstra_path(G, (233,8),(383,814)):
        image[idx] = 1
    im[image == 1] = 255, 0, 0
    plt.imshow(im)
    plt.show()

def main():
    #tau_fm= fast_marching()
    #maze_solver_fmm(tau_fm)
    #maze_solver_dij()


if __name__ == "__main__":
    main()