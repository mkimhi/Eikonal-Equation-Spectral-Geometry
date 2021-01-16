from MeshUtilites import Mesh
import numpy as np
import meshio
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import inv as sparse_inv
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
import matplotlib
import scipy
import matplotlib.colors as mcolors
from sklearn.manifold import MDS


def weighted_adjacency(v,f,cls='half_cotangent'):
    """
    Create a Weighted Adjacency matrix from a triangular mesh defined by (v,f).
    Two possible types:
    cls = "uniform": 1 for an exiting edge 0 otherwise
    cls = "half_cotangent" implement according to Meyer et al (2012):
        0.5(cot(alpha_ij)+ cot(beta_ij)) where alpha_ij,beta_ij are the far angles from the connected edge (i,j)
    :param v: vertices set
    :param f: faces set
    :param cls: "half_cotangent" (default), "uniform"
    :return: W
    """

    mesh = Mesh(v=v,f=f)

    if cls == 'half_cotangent':

        #Get Adj Matrices
        vertex_vertex_adj = mesh.vertex_vertex_adjacency()
        vertex_face_adj = mesh.vertex_face_adjacency()

        #List of all connected vertices
        connected_vertecis = [(i, j) for i, i_connected in
                              enumerate(np.split(vertex_vertex_adj.indices, vertex_vertex_adj.indptr)[1:-1])
                              for j in i_connected]
        vertex_face_adj_full = vertex_face_adj.toarray()
        data = []; rows_idxs = []; cols_idxs = []
        for itr,connected_pair in enumerate(connected_vertecis):
            #Get all triangels of each vertex
            # v1_tris_idxs = np.split(vertex_face_adj.indices, vertex_face_adj.indptr)[connected_pair[0] + 1]
            # v2_tris_idxs = np.split(vertex_face_adj.indices, vertex_face_adj.indptr)[connected_pair[1] + 1]

            tri1 = vertex_face_adj_full[connected_pair[0],:]
            tri2 = vertex_face_adj_full[connected_pair[1],:]

            #Find shared Triangels
            # tri1 = np.zeros((1,len(f)))
            # tri1[0,v1_tris_idxs] = 1
            # tri2 = np.zeros((1, len(f)))
            # tri2[0,v2_tris_idxs] = 1
            #shared_tri_idxs = np.where(tri1*tri2 == 1)[1]
            shared_tri_idxs = np.where(tri1 * tri2 == 1)

            #Get Far Vertecies
            shared_tri_vertices = f[shared_tri_idxs, :]
            far_vertices_idxs = shared_tri_vertices[~np.isin(shared_tri_vertices, connected_pair)]
            far_vertices = v[far_vertices_idxs, :]

            #Get Angels
            X = v[connected_pair,:]
            Yi = far_vertices[0,:] - X
            Yi = Yi / np.linalg.norm(Yi, axis=1).reshape((2, 1))
            alpha = np.arccos(np.dot(Yi[0,:],Yi[1,:]))
            Yj = far_vertices[1, :] - X
            Yj = Yj / np.linalg.norm(Yj, axis=1).reshape((2, 1))
            beta = np.arccos(np.dot(Yj[0, :], Yj[1, :]))

            #Store Data
            data.append(0.5*(1/np.tan(alpha) + 1/np.tan(beta)))
            rows_idxs.append(connected_pair[0])
            cols_idxs.append(connected_pair[1])

        return csr_matrix((data, (rows_idxs, cols_idxs)), shape=(len(v), len(v)))

    elif cls == 'uniform':
        return mesh.vertex_vertex_adjacency()
    else:
        raise NameError('No such Cls: {}'.format(cls))

def Laplacian(v,f,cls='half_cotangent',W=None):
    """
    Create a Laplacian matrix from adjaceny matrix W, L = D - W
    where D is the diagonal matrix contains the sum of each row of W
    :param v: vertices set
    :param f: faces set
    :param cls: "half_cotangent" (default), "uniform"
    :param W: a pre-calculated adjacency matrix instead of calculation a new one
    :return:
    """

    if W is None:
        W = weighted_adjacency(v,f,cls)
    D = diags(np.sum(W.toarray(), axis=1))
    L = D - W
    return L

def barycenter_vertex_mass_matrix(v,f):
    """
    Create a diagonal matrix with Barysentric Vertex Area of each vertex on its diagonal
    :param v: vertices set
    :param f: faces set
    :return:
    """
    mesh = Mesh(v=v, f=f)
    M = diags(mesh.barysentricVertexAreas())

    return M

def laplacian_spectrum(v,f,k,cls,W=None):
    """
    Generalized eigen value decomposition of the Laplace matrix and the Barycenter Vertex mass matrix taking the k
    samllest eigenvalues
    :param v:
    :param f:
    :param k:
    :param cls:
    :param W:
    :return:
    """
    L = Laplacian(v,f,cls,W=W)
    M = barycenter_vertex_mass_matrix(v,f)

    eig_val, eig_vec = eigsh(L, k, M, which='LM', sigma=0, tol=1e-7)
    eig_val = np.round(eig_val,12)
    eig_vec = np.round(eig_vec,12)

    return eig_val, eig_vec

def multi3DPloter(mesh_eig_tuple_arr,numOfEigFuncs= 1, renderType="Surface", ModelName=''):
    """
    Plot subplots of 3D mesh for given eigen functions
    :param mesh_eig_tuple_arr:
    :param numOfEigFuncs:
    :param renderType:
    :param ModelName:
    :return:
    """
    plotter = pv.Plotter(shape=(numOfEigFuncs, len(mesh_eig_tuple_arr)))
    cmap = plt.cm.get_cmap('jet')

    for eigFuncNum in range(numOfEigFuncs):
        for model_num in range(len(mesh_eig_tuple_arr)):
            eig_vec = mesh_eig_tuple_arr[model_num][1][:, eigFuncNum + 1]

            plotter.subplot(eigFuncNum, model_num)
            plotter.add_text("Model Name: "+ModelName+"{:03} Eigenfunction Number:{}".
                             format(model_num,eigFuncNum+1), font_size=8)

            if renderType == "PoinCloud":
                mesh = mesh_eig_tuple_arr[model_num][0]
                mesh = mesh.numpy_to_pyvista(mesh.v)
                mesh['scalar_func'] = eig_vec
                plotter.add_mesh(mesh, scalars='scalar_func', point_size=5.,
                                 render_points_as_spheres=True,cmap=cmap)
            elif renderType == "Surface":
                mesh = mesh_eig_tuple_arr[model_num][0]
                mesh = mesh.numpy_to_pyvista(mesh.v, mesh.f)
                mesh['scalar_func'] = eig_vec
                plotter.add_mesh(mesh, scalars='scalar_func', show_edges=False, cmap=cmap)
            else:
                raise NameError('No such renderType: {}'.format(renderType))

            plotter.view_xy()
            #plotter.add_scalar_bar("Eigenfunction Value")

    plotter.show()

def multi3DPloter2(mesh_list, scalar_func_mat_list, title_modelName_list = None,
                   title_funcName_List = None,cmap_type='jet',renderType="Surface",fontSize=8,saveModelName=None):

    #Check conssist number of function per model
    numOfFuncs_list = [scalar_func.shape[1] for scalar_func in scalar_func_mat_list]
    if len(set(numOfFuncs_list)) != 1:
        raise ValueError('In Consist Number Of Scalar Function Per Model')

    numOfFuncs = numOfFuncs_list[0]
    numOfModels = len(mesh_list)
    plotter = pv.Plotter(shape=(numOfModels,numOfFuncs))
    cmap = plt.cm.get_cmap(cmap_type)

    for mesh_num, model_mesh in enumerate(mesh_list):
        scalar_func_mat = scalar_func_mat_list[mesh_num]
        for scalar_func_num in range(numOfFuncs):
            plotter.subplot(mesh_num, scalar_func_num)
            if title_modelName_list is not None and title_funcName_List is not None:
                plotter.add_text("Model: {}, {}".
                                 format(title_modelName_list[mesh_num],
                                        title_funcName_List[scalar_func_num]), font_size=fontSize)

            if renderType == "PoinCloud":
                mesh = model_mesh.numpy_to_pyvista(model_mesh.v)
                mesh['scalar_func'] = scalar_func_mat[:,scalar_func_num]
                plotter.add_mesh(mesh, scalars='scalar_func', point_size=5.,
                                 render_points_as_spheres=True, cmap=cmap)
            elif renderType == "Surface":
                mesh = model_mesh.numpy_to_pyvista(model_mesh.v, model_mesh.f)
                mesh['scalar_func'] =  scalar_func_mat[:,scalar_func_num]
                plotter.add_mesh(mesh, scalars='scalar_func', show_edges=False, cmap=cmap)
            else:
                raise NameError('No such renderType: {}'.format(renderType))

            plotter.view_xy()
    if saveModelName is None:
        plotter.show()
    else:
        plotter.show(screenshot=saveModelName)

def multi2DScatterPlotter(MDS_dict_list,k_list,labels,numOfDecs = 4,figScale=1,title="",markerScale=5,saveFigName=""):
    numOfKs = len(k_list)

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    cmap = mcolors.ListedColormap(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    n = len(np.unique(labels))
    norm = mcolors.BoundaryNorm(np.arange(n + 1) - 0.5, n)
    ncol = numOfDecs
    nrow = numOfKs
    scaleFigX = figScale
    scaleFixY = figScale
    fig = plt.figure(figsize=(scaleFigX * (ncol + 1), scaleFixY * (nrow + 1)))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.1, hspace=0.1,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    fig.suptitle(title, fontsize="14")
    for i,k in enumerate(k_list):
        ax = plt.subplot(gs[i, 0])
        sct1 = ax.scatter(MDS_dict_list[i]['H'][:,0],MDS_dict_list[i]['H'][:,1], s=markerScale,c=labels,cmap=cmap,norm=norm)
        ax.grid()
        ax.set_ylabel(r"k={}".format(k), fontsize=14)
        if i == 0:
            ax.set_title(r"Mean Curvature Descriptor", fontsize=12)
        ax = plt.subplot(gs[i, 1])
        sct2 = ax.scatter(MDS_dict_list[i]['GPS'][:,0],MDS_dict_list[i]['GPS'][:,1], s=markerScale,c=labels,cmap=cmap,norm=norm)
        ax.grid()
        if i == 0:
            ax.set_title(r"GPS", fontsize=12)
        ax = plt.subplot(gs[i, 2])
        sct3 = ax.scatter(MDS_dict_list[i]['ShapeDNA'][:,0],MDS_dict_list[i]['ShapeDNA'][:,1], s=markerScale,c=labels,cmap=cmap,norm=norm)
        ax.grid()
        if i == 0:
            ax.set_title(r"DNA", fontsize=12)
        ax = plt.subplot(gs[i, 3])
        sct4 = ax.scatter(MDS_dict_list[i]['HKS'][:,0],MDS_dict_list[i]['HKS'][:,1], s=markerScale,c=labels,cmap=cmap,norm=norm)
        fig.colorbar(sct4, ticks=np.unique(labels))

        if i == 0:
            ax.set_title(r"HKS", fontsize=12)

        ax.grid()
    plt.show()
    if saveFigName:
        fig.savefig("{}.png".format(saveFigName))

def LaplacianDiscreteMeanCurveture(v,f,cls='half_cotangent',W=None):

    L = Laplacian(v,f,cls,W=W)
    M = barycenter_vertex_mass_matrix(v,f)
    Hn = sparse_inv(csc_matrix(M)) @ L @ v
    #Hn = sparse_inv(M) @ L @ v
    Hn_abs = np.linalg.norm(Hn,axis=1)

    #Get curveture sign by checking direction of curveture and the normal
    mesh = Mesh(v=v,f=f)
    normals = mesh.vertexNormals()
    Hn_normlized = Hn / Hn_abs.reshape((len(v),1))
    Hn_signs = np.sign(np.diag(Hn_normlized @ normals.T))

    Hn_scalarFunc = Hn_abs * Hn_signs


    return Hn_scalarFunc

def LBOAproximation(v,f,scalar_function,k,cls='half_cotangent'):

    eig_vals,eig_vects = laplacian_spectrum(v,f,k,cls)
    M = barycenter_vertex_mass_matrix(v,f)
    L = Laplacian(v,f,cls)

    scalar_function_hat = eig_vects @ eig_vects.T @ M @ scalar_function
    normlized_laplacian_scalarFunc = sparse_inv(csc_matrix(M)) @ L @ scalar_function

    return scalar_function_hat, normlized_laplacian_scalarFunc

def expspace(start, stop, n):
    return np.exp(np.linspace(np.log(start), np.log(stop), n))

def CreateSpectralDescriptor(mesh,descType,k=3,cls="half_cotangent",t=1.0,W=None):

    eig_val, eig_vec = laplacian_spectrum(v=mesh.v, f=mesh.f, k=k, cls=cls,W=W)

    if descType == "ShapeDNA":
        desc = eig_val[1:] # kx1

    elif descType == "GPS":
        desc = eig_vec[:,1:] @ np.diag(np.power(eig_val[1:],-0.5)) # |V|xk

    elif descType == "HKS":
        phi_squared = eig_vec ** 2
        exp_lambda_t = np.exp(-t * eig_val)
        desc = phi_squared @ exp_lambda_t # |V|x1

    elif descType =="All":
        phi_squared = eig_vec ** 2
        exp_lambda_t = np.exp(-t * eig_val)
        desc_HKS = phi_squared @ exp_lambda_t
        desc_ShapeDNA = eig_val[1:]
        desc_GPS = eig_vec[:, 1:] @ np.diag(np.power(eig_val[1:], -0.5))
        desc_GPS = desc_GPS.flatten() #TODO: Check if need to flat other way
        desc = {"HKS":desc_HKS, "GPS":desc_GPS, "ShapeDNA": desc_ShapeDNA}

    else:
        raise NameError("{} Is Not A Valid  descType".format(descType))

    return desc

def CreateDescriptorsPerwiseDistanceOfMatrix(base_path, model_nums_list, k, HKS_t=1e-4):

    #Create Multi Descriptor
    descriptors_names = ["H","HKS","GPS","ShapeDNA"]
    contacted_descriptors = {desc_name:[] for desc_name in descriptors_names}
    for model_num in model_nums_list:
        path = base_path + '{:03}'.format(model_num) + '.ply'
        mesh = Mesh(path)
        W = scipy.sparse.load_npz('tmp\\W_{}.npz'.format(model_num))
        try:
            multiDescriptor = CreateSpectralDescriptor(mesh,descType="All",k=k,cls="half_cotangent",t=HKS_t,W=W)
            multiDescriptor['H'] = LaplacianDiscreteMeanCurveture(mesh.v,mesh.f,cls='half_cotangent',W=W)
            for desc_name in descriptors_names:
                contacted_descriptors[desc_name].append(multiDescriptor[desc_name])
            print("Model:{} - DONE".format(model_num))
        except:
            print("Model:{} - Failed".format(model_num))

    contacted_descriptors = {desc_name: np.array(contacted_descriptors[desc_name]) for desc_name in descriptors_names}

    #Create Pairwise Distance Matrix By L2 norm
    distacne_matrix_dec_dict = {desc_name: pairwise_distances(contacted_descriptors[desc_name]) for desc_name in descriptors_names}

    return distacne_matrix_dec_dict, contacted_descriptors
def DistanceFromCenterOfMass(mesh):
    centeroid = np.mean(mesh.v, axis=0)
    distance_func = np.linalg.norm(mesh.v - centeroid, axis=1)
    return distance_func



def q2_main():
    #######################################################################
    # Q6 - Save Adjecny matrices - Done Once
    #######################################################################
    # base_path = 'MPI-FAUST\\training\\registrations\\tr_reg_'
    # model_num_range = range(100)
    #
    # for model_num in model_num_range:
    #     path = base_path + '{:03}'.format(model_num) + '.ply'
    #     mesh = Mesh(path)
    #     W = weighted_adjacency(v=mesh.v, f=mesh.f, cls='half_cotangent')
    #     scipy.sparse.save_npz('tmp\\W_{}.npz'.format(model_num), W)
    #     print("Adj Matrix For Model:{} - Saved".format(model_num))
    #######################################################################
    # Q6 - descriptors Compartion (Load Data)
    #######################################################################
    desc_k50 = np.load("descs_k50.npy", allow_pickle=True).item()
    desc_k200 = np.load("descs_k200.npy", allow_pickle=True).item()
    descs_list = [desc_k50, desc_k200]
    model_num_range = list(range(100))  # [0,10,20]
    model_num_range.remove(63)  # Fault ply
    numOfModelsPerClass = 10
    labels_subjects = [t // numOfModelsPerClass for t in model_num_range]
    labels_poses = [t % numOfModelsPerClass for t in model_num_range]
    descriptors_names = ["H", "HKS", "GPS", "ShapeDNA"]
    k_list = [50, 200, 1000]

    MDS_dict_list = []
    for i, k in enumerate(k_list):
        MDS_dict = {}
        if i == 2:  # for k=1000 saved already as MDS
            MDS_dict = np.load('MDS_k1000.npy', allow_pickle=True).item()
        else:
            for desc_name in descriptors_names:
                embedding = MDS(n_components=2)
                descs = descs_list[i]
                MDS_dict[desc_name] = embedding.fit_transform(descs[desc_name])
        MDS_dict_list.append(MDS_dict)
    multi2DScatterPlotter(MDS_dict_list, k_list, labels_subjects, figScale=1,
                          title="Descriptors MDS Colored By Subjects", markerScale=5)

    #######################################################################
    # Q6 - descriptors Compartion
    #######################################################################
    base_path = 'MPI-FAUST\\training\\registrations\\tr_reg_'
    k_list = [200]  # [50,200,1000]
    numOfModelsPerClass = 10
    model_num_range = list(range(100))  # [0,10,20]
    model_num_range.remove(63)  # Fault ply
    labels_subjects = [t // numOfModelsPerClass for t in model_num_range]
    labels_poses = [t % numOfModelsPerClass for t in model_num_range]
    labels = [t // numOfModelsPerClass for t in model_num_range]
    cls_type = 'half_cotangent'
    t = 1e-4
    descriptors_names = ["H", "HKS", "GPS", "ShapeDNA"]

    # MDS
    MDS_dict_list = []
    for k in k_list:
        distacne_matrix_dec_dict, descriptores_dict = CreateDescriptorsPerwiseDistanceOfMatrix(base_path,
                                                                                               model_num_range, k,
                                                                                               HKS_t=t)
        np.save('dist_mat_k{}.npy'.format(k), distacne_matrix_dec_dict)
        MDS_dict = {}
        for desc_name in descriptors_names:
            # MDS_dict[desc_name]= DiffusionMapsEmbedding(M=distacne_matrix_dec_dict[desc_name], n_dim=2, t=1, kernel_method="Gaussian",
            #                                             X=None,  n_nei=None, epsilon=None)
            embedding = MDS(n_components=2)
            MDS_dict[desc_name] = embedding.fit_transform(descriptores_dict[desc_name])

        MDS_dict_list.append(MDS_dict)
        print("K={} - DONE".format(k))

    multi2DScatterPlotter(MDS_dict_list, k_list, labels, figScale=1, title="Comparing Descriptors Using MDS On FASUT",
                          markerScale=5, saveFigName="MDS1.png")
    pass

    #######################################################################
    # Q6 - HKS Times Demo
    #######################################################################
    base_path = 'MPI-FAUST\\training\\registrations\\tr_reg_'
    k = 500
    model_num_range = range(3)
    cls_type = 'half_cotangent'
    start_time = 1e-4
    end_time = 1
    time_length = 10
    times = expspace(start_time, end_time, time_length)
    sclar_func_list = []
    model_list = []
    model_title_list = []
    func_name_list = ["HKS t={:.3e}".format(t) for t in times]

    for model_num in model_num_range:
        path = base_path + '{:03}'.format(model_num) + '.ply'
        mesh = Mesh(path)

        HKS_list = [CreateSpectralDescriptor(mesh, descType="HKS", k=k, cls="half_cotangent", t=t) for t in times]
        HKS_mat = np.array([hks for hks in HKS_list])
        HKS_mat = HKS_mat.T
        model_list.append(mesh)
        sclar_func_list.append(HKS_mat)
        model_title_list.append("{:03}".format(model_num))

    multi3DPloter2(model_list, sclar_func_list, title_modelName_list=model_title_list,
                   title_funcName_List=func_name_list,
                   cmap_type='jet', renderType="Surface", fontSize=12)
    #######################################################################
    # Q5 - Laplacian applications on various scalar functions
    #######################################################################
    base_path = 'MPI-FAUST\\training\\registrations\\tr_reg_'
    model_num = 0
    k_list = [10, 200]
    cls_type = 'half_cotangent'
    path = base_path + '{:03}'.format(model_num) + '.ply'
    mesh = Mesh(path)
    laplacian_signed_curveture = LaplacianDiscreteMeanCurveture(v=mesh.v, f=mesh.f, cls=cls_type)
    distance_from_center = DistanceFromCenterOfMass(mesh)
    distance_func_list = []
    curve_func_list = []
    for k in k_list:
        curve_hat, normlized_laplacian_curve = LBOAproximation(v=mesh.v, f=mesh.f,
                                                               scalar_function=laplacian_signed_curveture, k=k,
                                                               cls='half_cotangent')
        distance_hat, normlized_laplacian_distance = LBOAproximation(v=mesh.v, f=mesh.f,
                                                                     scalar_function=distance_from_center, k=k,
                                                                     cls='half_cotangent')
        distance_func_list.append(distance_hat)
        curve_func_list.append(curve_hat)

    distance_func_list.append(distance_from_center)
    curve_func_list.append(laplacian_signed_curveture)
    func_distance_mat = np.array([distance_func_list])[0, :, :].T
    func_curve_mat = np.array([curve_func_list])[0, :, :].T

    # Normlized Curvature colors for visualtization
    mean = func_curve_mat.mean(axis=0)
    std = func_curve_mat.std(axis=0)
    func_curve_mat = np.clip(func_curve_mat, -std - mean, std + mean)
    func_curve_mat -= mean

    scalar_func_mat_list = [func_distance_mat, func_curve_mat]

    multi3DPloter2([mesh, mesh], scalar_func_mat_list, title_modelName_list=None,
                   title_funcName_List=None, cmap_type='jet', renderType="Surface", fontSize=8, saveModelName=None)

    #######################################################################
    # Q4 - Laplacian and the discrete Mean Curvature:
    #######################################################################
    base_path = 'MPI-FAUST\\training\\registrations\\tr_reg_'
    model_num = 12
    cls_type = 'half_cotangent'
    path = base_path + '{:03}'.format(model_num) + '.ply'
    model_mesh = Mesh(path)
    laplacian_signed_curveture = LaplacianDiscreteMeanCurveture(v=model_mesh.v, f=model_mesh.f, cls=cls_type)
    mesh = model_mesh.numpy_to_pyvista(model_mesh.v, model_mesh.f)
    mesh['scalar_func'] = laplacian_signed_curveture
    plotter = pv.Plotter()
    cmap = plt.cm.get_cmap('jet')
    plotter.add_mesh(mesh, scalars='scalar_func', show_edges=False, cmap=cmap)
    plotter.view_xy()
    plotter.show()
    pass

    #######################################################################
    # Q3 - Isometry invariance of the Laplacian
    #######################################################################
    numOfModels = 5
    numOfEigs = 6
    cls_type = 'half_cotangent'
    base_path = 'MPI-FAUST\\training\\registrations\\tr_reg_'
    mesh_eig_tupels_arr = []
    for model_num in range(numOfModels):
        path = base_path + '{:03}'.format(model_num) + '.ply'
        mesh = Mesh(path)
        eig_vals, eig_vec = laplacian_spectrum(v=mesh.v, f=mesh.f,
                                               k=numOfEigs, cls=cls_type)
        mesh_eig_tupels_arr.append((mesh, eig_vec))

    multi3DPloter(mesh_eig_tupels_arr, numOfEigFuncs=numOfEigs - 1, renderType="Surface")

    #######################################################################
    # Q2 - Check Laplacin Spectrum
    #######################################################################
    path = 'tr_reg_000.ply'
    mesh = Mesh(path)
    eig_vals, eig_vec = laplacian_spectrum(mesh.v, mesh.f, 3, cls='half_cotangent')
    mesh.render_pointcloud(scalar_func=eig_vec[:, 1])
    pass