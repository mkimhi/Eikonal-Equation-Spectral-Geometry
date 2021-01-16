import numpy as np
import itertools
import scipy.sparse
import pyvista as pv
import matplotlib.pyplot as plt
import meshio

def read_off(path):
    """
    Get a path to .off file and returns a tuple (v,f) where v is a list of vertices and f is a list of faces as
     described in Shared Vertex Data structure (Reaction 1)
    :param path: string for .off file
    :return: tuple(v,f)
    """

    f = open(path, 'r')
    # Get Number of Vertcies and Faces
    sizes = np.loadtxt(itertools.islice(f, 2), delimiter=' ', skiprows=1, dtype=int)
    numOfVertecis = sizes[0]
    numOfFaces = sizes[1]

    # Get Vertices
    #vertices = np.loadtxt(itertools.islice(f, 0, numOfVertecis), delimiter=' ', skiprows=0, dtype=float)
    vertices = np.loadtxt(itertools.islice(f, 0, numOfVertecis), skiprows=0, dtype=float)

    # Get Faces
    faces = np.loadtxt(itertools.islice(f, 0, numOfFaces), skiprows=0, dtype=int)[:, 1:]

    return vertices, faces

def write_off(path, vertices_faces_tuple):
    """
    Write an .off file to location path with the vertices and faces mentioned in vertices_faces_tuple
    :param path:
    :param vertices_faces_tuple:
    :return:
    """

    # Get Sizes
    numOfVertecis = vertices_faces_tuple[0].shape[0]
    numOfFaces, meshDim = vertices_faces_tuple[1].shape

    sizes = np.array([[numOfVertecis, numOfFaces, 0]])
    f = open(path, 'w')
    np.savetxt(f, sizes, delimiter=' ', newline='\n', fmt='%d', header='OFF', comments='')
    np.savetxt(f, vertices_faces_tuple[0], delimiter=' ', newline='\n', fmt='%f')
    np.savetxt(f, np.concatenate((meshDim * np.ones(shape=(numOfFaces, 1)), vertices_faces_tuple[1]), axis=1),
               delimiter=' ', newline='\n', fmt='%d')
    f.close()
    pass

# Basic Mesh Class
class Mesh:
    def __init__(self, path ='',v= None,f=None):
        if v is not None and f is not None:
            self.v = v
            self.f = f
        else:
            m = meshio.Mesh.read(path)
            self.v = m.points
            self.f = m.cells_dict['triangle']

        #self.v, self.f = read_off(path)


    def vertex_face_adjacency(self):
        """
        Retrun a sparse Boolean vertex-face adjacency matrix of size |V|x|F|
        :return:
        """
        # Sizes
        V = np.shape(self.v)[0]
        F = np.shape(self.f)[0]

        A = np.zeros(shape=(V, F))

        for idx, faces in enumerate(self.f):
            A[faces.flatten(), idx] = 1

        return scipy.sparse.csr_matrix(A)

    def vertex_vertex_adjacency(self):
        """
        returns the sparse Boolean vertex-vertex adjacency matrix of size |V|x|V|
        :return:
        """

        V = np.shape(self.v)[0]
        A_vf = self.vertex_face_adjacency()

        A = A_vf * A_vf.transpose()
        A_vv = A.astype(dtype=bool) - scipy.sparse.identity(V)

        return A_vv

    def vertex_degree(self):
        """
        Returns a vector of each vertex degree
        :return:
        """

        A_vv = self.vertex_vertex_adjacency()

        return A_vv.sum(axis=1)

    def numpy_to_pyvista(self, v, f=None):
        """
        Utility function to create a mesh compatible with Pyvista
        :param f:
        :return:
        """
        if f is None:
            return pv.PolyData(v)
        else:
            return pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))

    def render_wireframe(self):
        """
        Cretea a Wireframe of the mesh using Pyvista tools
        :return:
        """

        mesh = self.numpy_to_pyvista(v=self.v, f=self.f)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='wireframe', show_edges=True)
        plotter.view_xy()
        plotter.add_axes()

        plotter.show(window_size=[800, 600])

    def render_pointcloud(self, scalar_func, stitle='Scalar Function', visualize=True, cmap_name= "jet"):
        """
        Paint vetrices by scalr function
        :return:
        """
        mesh = self.numpy_to_pyvista(v=self.v)
        mesh['scalar_func'] = scalar_func

        cmap = plt.cm.get_cmap(cmap_name)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='scalar_func', point_size=5.,
                         render_points_as_spheres=True, cmap=cmap, stitle=stitle)

        plotter.view_xy()


        if visualize:
            plotter.show(window_size=[800, 600])

        return mesh

    def render_surface(self, scalar_func, stitle='Scalar Function', visualize=True):
        """
        Render the mesh surface with faces scalar function
        :return:
        """

        mesh = self.numpy_to_pyvista(v=self.v, f=self.f)
        mesh['scalar_func'] = scalar_func
        cmap = plt.cm.get_cmap("jet")

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='scalar_func', show_edges=True, cmap=cmap, stitle=stitle)

        plotter.view_xy()
        plotter.add_axes()

        if visualize:
            plotter.show(window_size=[800, 600])

        return mesh

    def faceNormals(self, normlizeNorm=True, visualize=False):
        """
        Get all faces normals by calculating the cross products of two edges.
        The normals are pointed out.
        By defult the normals are with unit size otherwise use normlizeNorm=False
        :param normlizeNorm:
        :return:
        """

        e1 = self.v[self.f[:, 1]] - self.v[self.f[:, 0]]
        e2 = self.v[self.f[:, 2]] - self.v[self.f[:, 0]]
        N = np.cross(e1, e2)

        if normlizeNorm:
            N = N / np.reshape(np.linalg.norm(N, axis=1), (self.f.shape[0], 1))

        if visualize:
            mesh = self.numpy_to_pyvista(self.v, self.f)
            p = pv.Plotter()
            p.add_arrows(self.faceBarycenters()[range(1, self.f.shape[0], 50)], N[range(1, self.f.shape[0], 50)],
                         mag=0.01)
            p.add_mesh(mesh, show_edges=True)
            p.show()

        return N

    def render_normals(self, normlized_Normals=True, normal_type='vertex', mag=0.01, presention_skip_frac=0.01,
                       base_mesh=None,stitle=None):
        """
        Render vertex \ face normals acording to normal_type.
        normlized_Normals - controsl whether the normals are normlized or not
        mag - control the size of the visualied arrow

        :param normal_type: 'vertex' or 'face'
        :param mag:
        :return:
        """

        if base_mesh is None:
            mesh = self.numpy_to_pyvista(self.v, self.f)
        else:
            mesh = base_mesh

        p = pv.Plotter()


        if normal_type == 'vertex':
            N = self.vertexNormals(normlizeNorm=normlized_Normals)
            centers = self.v
            skip_size = int(self.v.shape[0] * presention_skip_frac)
            p.add_arrows(centers[range(0, self.v.shape[0], skip_size)], N[range(0, self.v.shape[0], skip_size)],
                         mag=mag)
            p.add_mesh(mesh, show_edges=True,stitle=stitle,render_points_as_spheres=True)
            p.view_xy()

        elif normal_type == 'face':
            N = self.faceNormals(normlizeNorm=normlized_Normals)
            centers = self.faceBarycenters()
            skip_size = int(self.f.shape[0] * presention_skip_frac)
            p.add_mesh(mesh, show_edges=True,stitle=stitle)
            p.add_arrows(centers[range(0, self.f.shape[0], skip_size)], N[range(0, self.f.shape[0], skip_size)],
                         mag=mag)
            p.view_xy()

        p.show()

    def faceBarycenters(self):
        """
        Returns the Barycenters of each face
        :return:
        """

        return np.mean(self.v[self.f], axis=1)

    def faceAreas(self):
        """
        Return all faces areas using Heron's formula.
        Given triangle side lengths a,b,c and s = 0.5*(a+b+c) then A = sqrt(s*(s-a)*(s-b)*(s-c))
        :return:
        """

        a = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 1]], axis=1)
        b = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 2]], axis=1)
        c = np.linalg.norm(self.v[self.f[:, 1]] - self.v[self.f[:, 2]], axis=1)

        s = (a + b + c) * 0.5

        A = np.sqrt(s * (s - a) * (s - b) * (s - c))

        return A

    def barysentricVertexAreas(self):
        """
        Returns the area of each vertex defined by the one third of all immediately adjacent traingles area.
        :return:
        """

        vetrcies_areas_list = []
        A_vf = self.vertex_face_adjacency().todense()
        areas = self.faceAreas().transpose()

        for idx, v in enumerate(self.v):
            aa = areas[np.array(A_vf[idx, :] == 1).flatten()]
            vetrcies_areas_list.append(np.sum(aa) / 3)

        return np.array(vetrcies_areas_list)

    def vertexNormals(self, normlizeNorm=True):
        """
        Compute for each vertex its normal as weighted sum over adjacent face normals
        :param normlizeNorm:
        :return:
        """
        N_vertcies_list = []
        A_vf = self.vertex_face_adjacency().todense()
        areas = self.faceAreas()
        face_Normals = self.faceNormals(normlizeNorm=True)

        for idx, v in enumerate(self.v):

            areas_f = areas[np.array(A_vf[idx, :]).astype(bool).flatten()]
            normals_f = face_Normals[np.array(A_vf[idx, :]).astype(bool).flatten()]

            n_v = normals_f.transpose().dot(areas_f.transpose()).transpose()

            if normlizeNorm:
                n_v = n_v / np.linalg.norm(n_v)

            N_vertcies_list.append(n_v)

        return np.array(N_vertcies_list)

    def gaussianCurvature(self):
        """
        Compute the gaussian curvature at every vertex using Angle Defect formulation
        :return:
        """

        A_v = self.vertex_face_adjacency().todense()
        vertex_areas = self.barysentricVertexAreas()
        gaussian_curveture_list = []

        for idx, v in enumerate(self.v):
            # Get all faces sides vectors
            t = self.v[self.f[np.array(A_v[idx, :]).flatten().astype(bool)]] - v
            # Remove the tested vertex (zero vector)
            #t = t[t != np.zeros((1, 3))].reshape([t.shape[0], t.shape[1] - 1, t.shape[2]])
            z = np.reshape(t,[t.shape[0]*t.shape[1],t.shape[2]])
            t = np.reshape(z[np.where(z.any(axis=1)),:],[t.shape[0], t.shape[1] - 1, t.shape[2]])
            # tt = tt.reshape([t.shape[0], t.shape[1] - 1, t.shape[2]])

            # Get matrix of all 2 sides of the ray starts at v
            e1 = t[:, 0, :]
            e2 = t[:, 1, :]
            # Get ray angle on vetex v
            e1 = e1 / np.linalg.norm(e1, axis=1).reshape([e1.shape[0], 1])
            e2 = e2 / np.linalg.norm(e2, axis=1).reshape([e2.shape[0], 1])

            angles_v = np.arccos(np.sum(e1 * e2, axis=1))

            K_v = (2 * np.pi - np.sum(angles_v)) / vertex_areas[idx]

            gaussian_curveture_list.append(K_v)

        # a = np.array(gaussian_curveture_list)
        # tst = np.sum(a * vertex_areas)

        return np.array(gaussian_curveture_list)

