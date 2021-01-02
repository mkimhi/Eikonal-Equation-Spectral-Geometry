import numpy as np
import matplotlib.pyplot as plt
from Mesh import Mesh
import eikonalfm
import cv2

"""
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import eye as sparse_eye
from matplotlib import gridspec
import matplotlib
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
"""


def put_forground_on_background(background, forground, x=0, y=0):
    back = background.copy()
    rows, cols, channels = forground.shape
    trans_indices = forground[...,2] != 0 # Where not transparent
    overlay_copy = back[y:y+rows, x:x+cols]
    overlay_copy[trans_indices] = forground[trans_indices]
    back[y:y+rows, x:x+cols] = overlay_copy
    return back

def neg_grad(gx,gy,cur):
    directions={}
    """directions[(cur[0],cur[1]+1)] = gy[cur[0],cur[1]+1] #up
    directions[cur[0], cur[1] - 1] = gy[cur[0], cur[1] - 1] #down
    directions[cur[0]+1, cur[1]] = gy[cur[0]+1, cur[1]] #right
    directions[cur[0]-1, cur[1]] = gy[cur[0]-1, cur[1]] #left"""
    directions[(cur[0],cur[1]+1)] = gx[cur]+gy[cur] #up
    directions[cur[0], cur[1] - 1] = gy[cur[0], cur[1] - 1] #down
    directions[cur[0]+1, cur[1]] = gy[cur[0]+1, cur[1]] #right
    directions[cur[0]-1, cur[1]] = gy[cur[0]-1, cur[1]] #left
    return min(directions, key=directions.get)


def main():
    maze_image = cv2.imread("maze.png")
    c= cv2.cvtColor(maze_image,cv2.COLOR_BGR2GRAY)

    #plt.imshow(c,cmap='gray')
    #plt.show()
    c[c>100] = 255
    c[c<101]=1

    x_s = (383,814)
    dx = (1.0, 1.0)
    order = 2
    tau_fm = eikonalfm.fast_marching(c, x_s, dx, order)
    #tau1_ffm = eikonalfm.factored_fast_marching(c, x_s, dx, order)
    #tau0 = eikonalfm.distance(tau1_ffm.shape, dx, x_s, indexing="ij")

    #plt.contourf(tau_fm)
    #plt.show()

    gx, gy = np.gradient(tau_fm)
    cur = (383,814)
    target = (233,8)
    solve_maze =maze_image.copy()
    i=0
    while (cur!=target):
        solve_maze[cur] = [1,0,0]
        cur = neg_grad(gx,gy,cur)
        i+=1
        print(cur)
        if (i==10000):
            break
    plt.imshow(solve_maze)
    plt.show()


if __name__ == "__main__":
    main()