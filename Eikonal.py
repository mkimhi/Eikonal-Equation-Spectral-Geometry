import numpy as np
import matplotlib.pyplot as plt
import eikonalfm
import networkx as nx
from PIL import Image
import scipy.io as sio
import cv2
import gdist


"""
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import eye as sparse_eye
from matplotlib import gridspec
import matplotlib
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
"""


#part 1
def fast_marching(image,source,show= False,maze=True):
    if maze:
        wall_weight,path_weight = 1000,1
        image[image == 1] = wall_weight
        image[image < 1] = path_weight

    tau_fm_maze = eikonalfm.fast_marching(image, source, (1.0, 1.0), 2) #dx and order
    #tau0_maze = eikonalfm.distance(tau_fm_maze.shape, (1.0, 1.0), source_point, indexing="ij")
    if show:
        plt.imshow(tau_fm_maze,cmap="jet")
        plt.colorbar()
        plt.show()
    return tau_fm_maze

def maze_solver_fmm(im_mame,source,target,show=True,maze=True):
    im=np.array(Image.open(im_mame))#''))
    c = np.asanyarray(Image.open(im_mame).convert('1')).astype('double')
    image =np.zeros(c.shape)

    tau_fm = fast_marching(c,source = np.array(source),show=False,maze)

    grad_x, grad_y = np.gradient(tau_fm)
    location = target
    for i in range(100000):
        if location[0] == source[0] and location[1] == source[1] :
            print("got to source in: ",i)
            break
        image[location[0]-1:location[0]+1, location[1]-1: location[1]+1] = 1
        grad = np.array([grad_x[location[0], location[1]], grad_y[location[0], location[1]]])
        grad_abs = np.abs(grad)
        if grad_abs[0] < 0.0001:
            grad[0] = 0
        elif grad_abs[1] < 0.0001:
            grad[1] = 0
        grad_abs[grad_abs == 0] = 1 #not devide by 0
        location -= (grad / grad_abs).astype(int)
        if i > 90000:
            print(grad)
    im[image==1] = 255,0,0
    if show:
        plt.imshow(im,cmap="jet")
        plt.colorbar()
        plt.show()
    return i

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


#part 2
def Optical_Path_Length():
    pool = sio.loadmat("pool.mat")['n']
    pool[pool > 1.01] =5
    pool=  1/ pool#**10
    in_pool = np.array([499,399])
    out_pool = np.array([0,0])
    dx = (1., 1.)
    order = 2
    tau_fm_pool = eikonalfm.fast_marching(pool, in_pool, dx, order)

    #plt.imshow(pool, cmap="jet")
    #plt.colorbar()
    #plt.show()
    image = np.zeros(pool.shape)
    grad_x, grad_y = np.gradient(tau_fm_pool)
    location = out_pool
    im = sio.loadmat("pool.mat")['n']

    for _ in range(100000):
        if location[0] == in_pool[0] and location[1] == in_pool[1]:
            break
        if location[0]==500:
            break
        image[location[0], location[1]] = 1
        grad = np.array([grad_x[location[0], location[1]], grad_y[location[0], location[1]]])
        grad_abs = np.abs(grad)
        grad_abs[grad_abs == 0] = 1 #not to devide by 0
        location -= (grad / grad_abs).astype(int)
    im[image==1] = 1.1
    plt.imshow(im, cmap="jet")
    plt.colorbar()
    plt.show()


#part 3

def segmentation_0(I):
    canny  = cv2.Canny(I, 10, 100)
    plt.imshow(canny)
    plt.show()

def segmentation(I,sigma=1,epsilon = 1e-3):
    h,w = I.shape[:2]
    #p is a list of tuples of 4 corners of the object: (top left) (top right) (bottom right) (bottom left)
    p = [np.array([0, 0]),np.array([0, w]),np.array([h, w]),np.array([h, 0])]


    """#promotes geodesics that pass along edges of the image
    # todo: standard deviation of K is sigma
    K = 0 #gaus kernel #todo
    grad_K =np.gradient(K)
    g = 1/ (1+np.linalg.norm(np.convolve(grad_K, I) ))#todo: Canny edge detector instead of the norm in denometer"""
    canny = cv2.Canny(I, 10, 100)
    g = 1 / (1+canny)

    energy = 0 #todo :sum(i=0-3) over integrate p_i to p_j(means?) over G
    prev_energy = energy + 1

    while (energy - prev_energy > epsilon):
        prev_energy = energy

        tau_fm_maze = eikonalfm.fast_marching(g, p[0], (1.0, 1.0), 2)



        #todo: the algo - fucking rythem
        """Compute the geodesic between every adjacent pair of points(pixels) i,j
        (4 geodesics, where j = (i+1) mod 4)  and find the midway points q[0:3]
        along each path where the distance functions are the same (D(pi) = D(pj)) where D=integral_pi_pj on g 
        set p=q"""
        #epsilon_t WITH P=Q