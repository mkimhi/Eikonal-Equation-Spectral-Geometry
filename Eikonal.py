import numpy as np
import matplotlib.pyplot as plt
import eikonalfm
import networkx as nx
from PIL import Image
import scipy.io as sio
import cv2
import gdist
import meshio
from meshio import read
import scipy
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from Mesh import Mesh




#part 1
def fast_marching(image,source, show= False,maze=True):
    """
    create a matrix of FMM fo image from source
    :param image:
    :param source:
    :param show:
    :param maze:
    :return:
    """
    if maze:
        wall_weight,path_weight = 100,1
        image[image == 1] = wall_weight
        image[image < 1] = path_weight

    tau_fm_maze = eikonalfm.fast_marching(image, source, (1.0, 1.0), 2) #dx and order
    #tau0_maze = eikonalfm.distance(tau_fm_maze.shape, (1.0, 1.0), source_point, indexing="ij")
    if show:
        plt.imshow(tau_fm_maze,cmap="jet")
        plt.colorbar()
        plt.show()
    return tau_fm_maze
def maze_solver_fmm(im,source,target,show=True,maze=True,steps = 100000):
    """

    :param im: PIL image
    :param source: source to create the FMM from
    :param target: the target point we start the maze/walk from
    :param show: if True, show the solve of the maze
    :param maze: if True, we do the maze adaption of thw walls for the image (walls-1000,walk-1)
    :param steps: if you want to calculate the point in a midway, insert steps and return the index of the
    point at steps distance from the target to source
    :return: the geodestic distance and location of target/midway point, also the trajectory
    """
    im = np.array(im)
    binary_im = np.asanyarray(Image.fromarray(im).convert('1')).astype('double')
    zero_image = np.zeros(binary_im.shape)

    tau_fm = fast_marching(binary_im,source = np.array(source),show=True,maze=maze)

    grad_x, grad_y = np.gradient(tau_fm)
    location = target.copy()
    traj=[]
    for i in range(steps):
        traj.append(location.copy())
        if location[0] == source[0] and location[1] == source[1] :
            break
        elif i == steps-1:
            break
            #return i, location, traj
        zero_image[location[0]-1:location[0]+1, location[1]-1: location[1]+1] = 1
        grad = np.array([grad_x[location[0], location[1]], grad_y[location[0], location[1]]])
        grad_abs = np.abs(grad)
        if grad_abs[0] < 0.0001:
            grad[0] = 0
        elif grad_abs[1] < 0.0001:
            grad[1] = 0
        grad_abs[grad_abs == 0] = 1 #not devide by 0
        location -= (grad / grad_abs).astype(int)
        #if i > 90000:
        #    print(grad)
    im[zero_image==1] = 255,0,255
    if show:
        plt.imshow(im,cmap="jet")
        plt.colorbar()
        plt.show()
    return i,location,traj
def maze_solver_dij():
    """
    dijkstra algorithm for solving maze
    :return:
    """
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
    """
    Optical_Path_Length to find shortest path to bottom of the pool
    :return:
    """
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
        image[location[0]-1:location[0]+1, location[1]-1:location[1]+1] = 1
        grad = np.array([grad_x[location[0], location[1]], grad_y[location[0], location[1]]])
        grad_abs = np.abs(grad)
        grad_abs[grad_abs == 0] = 1 #not to devide by 0
        location -= (grad / grad_abs).astype(int)
    im[image==1] = 1.1
    plt.imshow(im, cmap="jet")
    plt.colorbar()
    plt.show()

#part 3

def segmentation_old(I,sigma=1,epsilon = 2):
    """
    segment input image I
    :param I:
    :param sigma:
    :param epsilon:
    :return:
    """
    h,w = I.shape[:2]
    #p is a list of tuples of 4 corners of the object: (top left) (top right) (bottom right) (bottom left)
    p = [np.array([10, 10]),np.array([10, w-10]),np.array([h-10, w-10]),np.array([h-10, 10])]
    q = [0,0,0,0]

    canny = cv2.Canny(I*255, 10, 150)
    #plt.imshow(canny, cmap="jet")
    #plt.show()
    canny[canny==0]=1
    canny[canny ==255]=10000
    g = 1 / (1+canny)

    dist = [0,0,0,0] #p0 from p1, p1 from p2, p2 from p3, p3 from p0
    energy = 0 #todo :sum(i=0-3) over integrate p_i to p_j(means?) over G
    prev_energy = energy -5
    t = [[0], [0], [0], [0]]
    while (np.abs(energy - prev_energy) > epsilon):
        prev_energy = energy

        dist[0], _,t[0] = maze_solver_fmm(I, source=p[1], target=p[0], show=True, maze=True, steps=10000)
        dist[1], _,t[1] = maze_solver_fmm(I, source=p[2], target=p[1], show=False, maze=True, steps=10000)
        dist[2], _,t[2] = maze_solver_fmm(I, source=p[3], target=p[2], show=False, maze=True, steps=10000)
        dist[3],_ ,t[3]= maze_solver_fmm(I, source=p[0], target=p[3], show=False, maze=True, steps=10000)

        q[0] = t[0][len(t[0]) // 2]
        q[1] = t[1][len(t[1]) // 2]
        q[2] = t[2][len(t[2]) // 2]
        q[3] = t[3][len(t[3]) // 2]

        e1 = np.sum([g[i] for i in t[0]])
        e2 = np.sum([g[i] for i in t[1]])
        e3 = np.sum([g[i] for i in t[2]])
        e4 = np.sum([g[i] for i in t[3]])
        energy = e1+e2+e3+e4
        print(energy)
        #print(p)
        p=q
def geo_dist(domain, source, target):
    """
    calc deo distances
    :param domain:
    :param source:
    :param target:
    :return:
    """
    fmm_result = eikonalfm.fast_marching(domain, target, (1.0, 1.0), 2)
    fmm_grad = np.gradient(fmm_result)
    fmm_grad = list(map(lambda data: np.expand_dims(data, -1), fmm_grad))
    fmm_grad = np.concatenate(fmm_grad, axis=-1)
    curr = source.copy()
    locations = []
    last_location = None
    itr = 0

    while np.linalg.norm(curr - target) > 1 and itr < 3000:
        itr += 1
        curr[0] = np.clip(curr[0], 0, domain.shape[0] - 1)
        curr[1] = np.clip(curr[1], 0, domain.shape[1] - 1)
        int_location = curr.astype(int)
        fmm_result[int_location[0], int_location[1]] = 1000
        if np.any(last_location != int_location):
            last_location = int_location
            locations.append(int_location)
        grad = fmm_grad[int_location[0], int_location[1]]

        #make a step in an int size
        if np.linalg.norm(grad) !=0:# 1e-4:
            grad /= np.linalg.norm(grad)
        curr = curr - grad
    return locations
def segmentation(image, indexes=[0, 1,2,3 ,5, 9], sigma1=120,sigma2=190,dot_size=10):
    """
    a version that work of segmentation
    :param image:
    :param indexes:
    :param sigma1:
    :param sigma2:
    :param dot_size:
    :return:
    """
    line_color = [0, 150, 150]
    p_color = [200, 0, 200]
    q_color = [100, 200, 0]
    g = sum([cv2.Canny(image[..., i], sigma1,sigma2).astype(float) for i in range(3)])
    g = 1 / (1+g)

    h, w = image.shape[:2]
    p = [np.array([dot_size, dot_size]), np.array([dot_size, w - dot_size]), np.array([h - dot_size, w - dot_size]), np.array([h - dot_size, dot_size])]

    fig = plt.figure()
    fig.suptitle('iterations over segmentation')
    plt_idx = 1
    energies = []
    old_energy = 100
    for itr in range(10): #maximum iterations
        iter_image = image.copy()
        q = []
        energy = 0
        for i in range(len(p)):
            p_i = p[i]
            p_j = p[(i + 1) % len(p)]
            geodesic = geo_dist(g, p_i, p_j) #return the 4 points geodesic dist
            for point in geodesic:
                iter_image[point[0] - 3:point[0] + 3, point[1] - 3:point[1] + 3] = line_color
                energy += g[point[0], point[1]]

            q_i = geodesic[len(geodesic) // 2]
            #plot p and q dots:
            iter_image[p_i[0] - dot_size:p_i[0] + dot_size:, p_i[1] - dot_size:p_i[1] + dot_size] = p_color
            iter_image[q_i[0] - dot_size:q_i[0] + dot_size:, q_i[1] - dot_size:q_i[1] + dot_size] = q_color
            q.append(q_i)
        energies.append(energy)
        if (np.abs(energy - old_energy)<1e-1): #break when energy stop degredation
            break
        old_energy=energies[-1]
        p = q
        if itr in indexes:
            ax = fig.add_subplot(str(230+plt_idx))
            plt_idx += 1
            ax.imshow(iter_image)
    plt.show()
    plt.figure()
    plt.plot(energies)
    plt.title('Energy')
    plt.show()


#part 4
def embed_geodesic(meshes_names,n_dims=2):
    """
    create and save geodestics- run in colab and download files
    :param meshes_names:
    :param n_dims:
    :return:
    """
    for mesh_name in meshes_names:
        geodesic = scipy.sparse.load_npz(mesh_name+'.npz')
        embedding = MDS(n_dims=2)
        geodesic = geodesic.todense()
        MDS_geodestic = embedding.fit_transform(geodesic)
        spherical_MDS_gerdestic = np.cos(MDS_geodestic)
        scipy.sparse.save_npz(mesh_name + '_MDS' + '.npz', csr_matrix(MDS_geodestic))
        scipy.sparse.save_npz(mesh_name + '_sphere_MDS' + '.npz', csr_matrix(spherical_MDS_gerdestic))


def compute_errors(mesh_name, mds, mds_str='mds', embedded_dim=2, snap=True):
    """
    solve 1.4 to calc MDS and geodestic and return normed distance
    :param mesh_name:
    :param mds:
    :param mds_str:
    :param embedded_dim:
    :param snap:
    :return:
    """
    # Load mesh from .ply file
    ply = meshio.read(mesh_name + ".ply")
    v,f = ply.points,ply.cells_dict['triangle']

    if snap:
        geodesics_dist = scipy.sparse.load_npz(mesh_name + '.npz').todense()
    else:
        geodesics_dist = gdist.local_gdist_matrix(v.astype(np.float64), f.astype(np.int32))
        scipy.sparse.save_npz(mesh_name + '.npz', csr_matrix(geodesics_dist))
    # to present original mesh
    """mesh_class = Mesh(v=vertices, f=faces)
    mesh_class.render_pointcloud(scalar_function=geodesics_dist[:],snap_name = path + " Mesh")"""

    if snap:
        if mds_str == 'mds':
            emb_coordinates = scipy.sparse.load_npz(mesh_name + '_MDS_NO_G.npz').todense()
            emb_coordinates = np.abs(emb_coordinates)
        # for sphere
        else:
            emb_coordinates = scipy.sparse.load_npz(mesh_name + '_SPHER_MDS_NO_G.npz').todense()
            emb_coordinates = np.abs(emb_coordinates)
            out = np.zeros((emb_coordinates.shape[0], 3))
            out[:, :2] = +emb_coordinates
            emb_coordinates = out
            embedded_geodesics_dist = scipy.sparse.load_npz(mesh_name + mds_str + '.npz').todense()
    else:
        emb_coordinates = mds(geodesics_dist, embedded_dim)
        embedded_geodesics_dist = gdist.local_gdist_matrix(np.array(emb_coordinates).astype(np.float64),
                                                           f.astype(np.int32))
        scipy.sparse.save_npz(mesh_name + mds_str + '.npz', csr_matrix(embedded_geodesics_dist))
    # present the mesh of embeded:
    """embedded_mesh = Mesh(v=emb_coordinates, f=ply.cells_dict['triangle'])
    embedded_mesh.render_pointcloud(scalar_function=np.sum(embedded_geodesics_dist,axis=1)/len(vertices), snap_name = path + " Mesh using " + method_str)"""
    err = np.linalg.norm(geodesics_dist - embedded_geodesics_dist)
    return err / len(v)

def spectral_embedding(data, out_dims):
    e_vlas, e_vecs = np.linalg.eig(data)
    evals_sorted_idx = e_vlas.argsort()
    e_vlas.sort()
    e_vecs = e_vecs[:, evals_sorted_idx]
    e_vlas , e_vecs= e_vlas[-out_dims:], e_vecs[:, -out_dims:]
    return e_vecs @ np.power(np.diag(e_vlas), 0.5)

def reg_mds(data, out_dims, **kwargs):
    n = data.shape[0]
    J = np.identity(n) - (1 / n) * np.ones_like(data)
    out = -0.5 * J * (data ** 2) * J
    mds_out = spectral_embedding(out, out_dims)
    return mds_out

def cannonical_shape_MDS(meshes_names):
    for mesh_name in meshes_names:
        err1 = compute_errors(mesh_name, reg_mds, 'mds', 3)
        print('normed error: {0}'.format(err1))


#   q5
def gen_f(s):
    """
    create f for new grath to create mesh
    :param s:
    :return:
    """
    f=[]
    f.append([0,len(s)-1,len(s)])
    f.append([0,1,len(s)])
    for i in range(len(s)-2):
        f.append([i,i+1,i+2])
    return f
def Farthest_Point_Sampling(mesh_name,n):
    """
    add the new farest point to set and print n out of all of mesh
    :param mesh_name:
    :param n:
    :return:
    """
    mesh = read(mesh_name + ".ply")
    v, f = (np.array(mesh.points), np.array(mesh.cells_dict['triangle'], dtype=np.int32))
    geodesics_dist = gdist.local_gdist_matrix(v.astype(np.float64), f.astype(np.int32))

    mesh = Mesh(v=v,f=f)
    mesh.render_pointcloud(scalar_function=mesh.gaussianCurvature())

    s=[]
    s.append(np.random.randint(0, len(v)))
    while (len(s)!=n):
        max_dist = 0
        selected_v = None
        for i,v_i in enumerate(v):
            min_by_s = np.inf
            for s_i in s: #get minimum ovver all s_i
                dist = geodesics_dist[s_i][v_i]
                if dist < min_by_s:
                    min_by_s = dist
            if min_by_s > max_dist:
                max_dist = min_by_s
                selected_v = v_i

        v = np.delete(v,selected_v) #dont iterate v over this node anymore
        s.append(selected_v)
    f_new=gen_f(s)
    mesh = Mesh(v=v[np.array(s)], f=f_new)
    mesh.render_pointcloud(scalar_function=mesh.gaussianCurvature())

def q1_main():
    # 1.1
    # solve a maze with fast mraching
    maze_solver_fmm(Image.open('maze.png'),[383,814],[233,8])
    #solve it with dijkastra
    maze_solver_dij()


    # 1.2
    Optical_Path_Length()
    # 1.3
    easy_images = [np.array(Image.open('ball.png')), np.array(Image.open('CORONA.png'))]
    images = [np.array(Image.open('duck.png')),np.array(Image.open('dog.png')) ]
    for im in easy_images:
        segmentation(im)
    meshes_names= ['tr_reg_000' , 'tr_reg_001']
    # 1.4
    #embed_geodesic(meshes_names)
    cannonical_shape_MDS(meshes_names)
    # 1.5
    hands_up = 'tr_reg_079'
    Farthest_Point_Sampling(hands_up,200)