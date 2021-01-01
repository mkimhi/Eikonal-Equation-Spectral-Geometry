import os
import numpy as np
import itertools
import scipy
import scipy.sparse
from scipy.sparse import *
import matplotlib.pyplot as plt
import meshio
import pyvista as pv
import glob


def read_off(path):
    """
    input is a path for a .off file
    return a tupple of v,f

    file=open(path,"r")
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces
    """
    mesh_data = meshio.off.read(path)
    return mesh_data.points, mesh_data.cells_dict["triangle"]

def write_off(v_f_tupple,filename, path='./'):
    (verts,faces)=np.array(v_f_tupple)
    file = open(path+filename+".off","w")
    file.write("OFF\n")
    file.write(str(len(verts)) + " "+str(len(faces))+ " 0\n")
    [file.write(str(ver[0])+' '+str(ver[1])+' '+str(ver[2])+'\n') for ver in verts]
    faces_len=len(faces[0])
    [file.write(str(faces_len)+" " +str(face[0])+' '+str(face[1])+' '+str(face[2])+'\n') for face in faces]
    file.close()


def numpy_to_pyvista(v, f=None):
    if f is None:
        return pv.PolyData(v)
    else:
        return pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))


class Mesh:
    def __init__(self, file_path):
        (self.v, self.f) = read_off(file_path)
        self.v = np.array(self.v)
        self.f = np.array(self.f)
        # save also the mesh as pyvista for functionality
        self.pv_mesh = numpy_to_pyvista(self.v, self.f)
        self.vf_adj_mat = self.vertex_face_adjacency().todense()
        self.areas = self.Face_Areas()
        self.cmap ="rainbow"# plt.cm.get_cmap("viridis", 5)

    def vertex_face_adjacency(self):
        """
        return |V|x|F| boolean matrix of first order adjacency
        between vertexes and faces
        """
        A = np.zeros((np.shape(self.v)[0], np.shape(self.f)[0]))
        for i in range(len(self.f)):
            A[self.f[i].flatten(), i] = 1
        return csr_matrix(A).astype(bool)

    def vertex_vertex_adjacency(self):
        """
        return |V|x|V| boolean simetric matrix
        where vertexes on the same face show as True
        """
        # the work with .astype is to be able to do matrix multiplications
        # and do be able to ignore vertexes that exist in more then one face
        # so the eye will eliminiate does from the final matrix
        vf_matrix = self.vertex_face_adjacency().astype(int)
        A = vf_matrix * np.transpose(vf_matrix).astype(bool)
        A = A.astype(int) - csr_matrix(np.eye(np.shape(A)[0]))
        return A.astype(bool)

    def vertex_degree(self):
        """
        return an array size on |V| with the degree of each V in verteces
        simple by sum each line of the vertex_vertex_adjacency matrix
        """
        vv_matrix = self.vertex_vertex_adjacency()
        return np.sum(vv_matrix, dtype=int, axis=1)

    def render_wireframe(self, snap_name='wireframe'):
        p = pv.Plotter()
        p.add_mesh(self.pv_mesh, show_edges=True, style='wireframe',stitle="wireframe")
        p.show(screenshot=snap_name)

    def render_pointcloud(self, scalar_function="no",point_size = 5.0, snap_name='point_cloud'):
        if scalar_function=="no":
            scalar_function = self.vertex_degree()
        mesh = self.pv_mesh
        mesh['scalar_function'] = scalar_function
        p = pv.Plotter()
        p.add_mesh(mesh, scalars="scalar_function", cmap=self.cmap,style='points',
                   point_size=point_size, render_points_as_spheres=True,stitle="point cloud")
        p.show(screenshot=snap_name)

    def render_surface(self, scalar_function="no", snap_name='surface'):
        if scalar_function=="no":
            scalar_function = self.vertex_degree()
        mesh = self.pv_mesh
        mesh.point_arrays['scalar_function'] = scalar_function
        mesh['scalar_function'] = scalar_function
        p = pv.Plotter()
        p.add_mesh(mesh, scalars="scalar_function", cmap=self.cmap,  show_edges=False,)#, render_points_as_spheres=True)
        p.show(screenshot=snap_name)



    def faces_normals(self, show=True, normlize=True, magnitude=0.05):
        if not normlize and magnitude < 1:
            magnitude = 500
        e1 = self.v[self.f[:, 1]] - self.v[self.f[:, 0]]
        e2 = self.v[self.f[:, 2]] - self.v[self.f[:, 0]]
        normals = np.cross(e1, e2)
        if normlize:
            normals = normals / np.reshape(np.linalg.norm(normals, axis=1), (self.f.shape[0], 1))
        if show:
            p = pv.Plotter()
            p.add_arrows(self.Face_Barycenters(), normals, mag=magnitude)
            p.show()
        return normals

    def Face_Barycenters(self):
        return np.mean(self.v[self.f], axis=1)

    def Face_Areas(self):
        a = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 1]], axis=1)
        b = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 2]], axis=1)
        c = np.linalg.norm(self.v[self.f[:, 1]] - self.v[self.f[:, 2]], axis=1)
        surface = (a + b + c) * 0.5
        return np.sqrt(surface * (surface - a) * (surface - b) * (surface - c))

    def Barysentric_Vertex_Areas(self):
        """
        Returns the area of each vertex
        defined by the one third of all immediately adjacent faces areas.
        todo: try to do efficient no for loop
        """
        vetrcies_areas = [np.sum(self.areas[np.array(self.vf_adj_mat[i, :] == 1).flatten()]) / 3 for i in
                          range(len(self.v))]
        return np.array(vetrcies_areas)

    def Vertex_Normals(self, normlize=True):
        """
        return a normal for each vertex as weighted sum over adjacent face normals
        todo: try to do efficient no for loop
        """
        face_Normals = self.faces_normals(show=False)
        norm_vertcies = np.zeros((len(self.v), 3))
        for idx, _ in enumerate(self.v):
            areas_faces = self.areas[np.array(self.vf_adj_mat[idx, :]).flatten()]
            normals_faces = face_Normals[np.array(self.vf_adj_mat[idx, :]).flatten()]
            norm_vertex = np.dot(normals_faces.T, areas_faces)
            norm_vertex /= np.linalg.norm(norm_vertex)
            norm_vertcies[idx] = norm_vertex
        return norm_vertcies

    def gaussianCurvature(self):
        """
        Compute the gaussian curveture at every vertex using Angle Defect formulaiton
        """
        vf_adj_mat = np.array(self.vf_adj_mat)
        vertex_areas = self.Barysentric_Vertex_Areas()
        gaussian_curveture = np.zeros(len(self.v))
        epsilon = 1e-10
        for i, v in enumerate(self.v):
            sub_v = self.v[self.f[vf_adj_mat[i].flatten()]] - v
            z = np.reshape(sub_v, [sub_v.shape[0] * sub_v.shape[1], sub_v.shape[2]])
            sub_v = np.reshape(z[np.where(z.any(axis=1)), :], [sub_v.shape[0], sub_v.shape[1] - 1, sub_v.shape[2]])
            e1, e2 = sub_v[:, 0, :], sub_v[:, 1, :]
            ##the norm should never be zero, but due to nomeric error it could be, so i added epsilon
            e1 = e1 / (np.linalg.norm(e1, axis=1).reshape([e1.shape[0], 1]) + epsilon)
            e2 = e2 / (np.linalg.norm(e2, axis=1).reshape([e2.shape[0], 1]) + epsilon)
            total_tethas_v = np.arccos(np.sum(e1 * e2, axis=1))
            K_v = (2 * np.pi - np.sum(total_tethas_v)) / vertex_areas[i]
            gaussian_curveture[i] = np.abs(K_v)
        return gaussian_curveture

    def render_center(self,point_size=5):
        center= np.mean(self.v, axis=0,keepdims=True)
        mesh=self.pv_mesh
        distances = np.linalg.norm(self.v - center,axis=1)
        mesh['scalar_function'] = distances
        p = pv.Plotter()
        p.add_mesh(mesh, scalars="scalar_function", cmap=self.cmap,point_size=point_size,  show_edges=True, render_points_as_spheres=True)
        p.show()



def q5_plot(corse = 0.1 ,len=10):
    u = np.arange(-1*len, len, corse)
    v = np.arange(-1*len, len, corse)
    u, v = np.meshgrid(u, v)
    denometer = u ** 2 + v ** 2 + 1
    x = 2 * u / denometer
    y = 2 * v / denometer
    z = (u ** 2 + v ** 2 - 1) / denometer
    grid = pv.StructuredGrid(x, y, z)
    grid.plot(screenshot="Q5")


if __name__ == "__main__":

    file_path = 'example_off_files/'
    paths = []
    for filename in sorted(glob.glob(os.path.join(file_path, '*.off'))):
        paths.append(filename)
    Meshes = [Mesh(path) for path in paths[0:3]]
    M=Meshes[0]
    M.render_wireframe()
    M.render_pointcloud()
    M.render_surface()
    M.faces_normals(normlize=True, magnitude=0.01)
    M.faces_normals(normlize=False, magnitude=700)
    scalar_function = M.gaussianCurvature()
    M.render_pointcloud(scalar_function=scalar_function,point_size=5)
    M.render_center(point_size=10)
    q5_plot()