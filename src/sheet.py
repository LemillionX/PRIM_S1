import tensorflow as tf
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import erf
import matplotlib.cm as cm
from termcolor import colored
import time


def indexTo1D(i,j, sizeX):
    return j*sizeX+i

def set_solid_boundary(u, v, sizeX, sizeY, b=0):
    new_u = tf.identity(u)
    new_v = tf.identity(v)
    if (tf.rank(new_u).numpy() < 2):
        new_u = tf.expand_dims(new_u, axis=1)
        new_v = tf.expand_dims(new_v, axis=1)

    mask = tf.logical_or(tf.logical_or(tf.math.equal(tf.range(sizeX*sizeY) % sizeX, 0), tf.math.equal(tf.range(sizeX*sizeY) % sizeX, sizeX-1)),
                     tf.logical_or(tf.math.equal(tf.range(sizeX*sizeY) // sizeX, 0), tf.math.equal(tf.range(sizeX*sizeY) // sizeX, sizeY-1)))
    indices = tf.where(mask)[:, 0]
    if (tf.rank(indices).numpy() < tf.rank(new_u).numpy() ):
        indices = tf.expand_dims(indices, axis=1)

    new_u = tf.tensor_scatter_nd_update(new_u, indices, tf.constant(b, shape=indices.shape, dtype=tf.float32))
    new_v = tf.tensor_scatter_nd_update(new_v, indices, tf.constant(b, shape=indices.shape, dtype=tf.float32))
    
    return new_u, new_v

def tensorToGrid(u, sizeX, sizeY):
    grid = np.zeros((sizeX, sizeY))
    for j in range(sizeY):
        for i in range(sizeX):
            grid[j,i] = u[i + sizeX*j]
    return grid

def tf_build_laplacian_matrix(sizeX, sizeY, a, b):
    diagonals = [b * tf.ones(sizeX * sizeY, dtype=tf.float32)]
    offsets = [0]

    if sizeX > 1:
        diagonals.append(a * tf.ones(sizeX * sizeY - 1, dtype=tf.float32))
        offsets.append(1)
        diagonals.append(a * tf.ones(sizeX * sizeY - 1, dtype=tf.float32))
        offsets.append(-1)

    if sizeY > 1:
        diagonals.append(a * tf.ones(sizeX * (sizeY - 1), dtype=tf.float32))
        offsets.append(sizeX)
        diagonals.append(a * tf.ones(sizeX * (sizeY - 1), dtype=tf.float32))
        offsets.append(-sizeX)

    laplacian_matrix = tf.sparse.SparseTensor(indices=tf.range(sizeX * sizeY, dtype=tf.int64),
                                              values=tf.concat(diagonals, axis=0),
                                              dense_shape=(sizeX * sizeY, sizeX * sizeY))
    laplacian_matrix = tf.sparse.add(laplacian_matrix, tf.sparse.eye(sizeX * sizeY, sizeX * sizeY, dtype=tf.float32))
    laplacian_matrix = tf.sparse.reorder(laplacian_matrix)

    return tf.linalg.lu(laplacian_matrix)

def build_laplacian_matrix(sizeX,sizeY,a,b):
    '''
    Build a Laplacian Matrix where the diagonal is full of ``b``, and adjacent cells are equals to ``a``. Can take a bit long to execute if the grid is large.
    
    Args:
        sizeX: An ``int`` representing the number of horizontal cells
        sizeY: An ``int`` representing the number of vertical cells
        a: A ``float`` for the value of the adjacent cells
        b: A ``float`` for the value of the diagonal

    Returns:
        The LU decomposition of the Laplacian Matrix, so that the linear solver is fast
    '''
    mat = np.zeros((sizeX*sizeY,sizeX*sizeY), dtype=np.float32)
    for it in range(sizeX*sizeY):
        i = it%sizeX
        j = it//sizeX
        if (i>0):
            mat[it,it-1] = a
            mat[it,it] += b/4
        if (i<sizeX-1):
            mat[it, it+1] = a
            mat[it, it] += b/4
        if (j>0):
            mat[it,it-sizeX] = a
            mat[it,it] += b/4
        if (j<sizeY-1):
            mat[it, it+sizeX] = a
            mat[it,it] += b/4
    sparse_mat = tf.sparse.from_dense(mat)
    tf.sparse.to_dense
    print(sparse_mat)
    tf.linalg.cholesky(sparse_mat)
    return tf.linalg.lu(sparse_mat)

def compute_divergence1(u, v, sizeX, sizeY, h):
    div = []
    for j in range(sizeY):
        for i in range(sizeX):
            s = 0
            if (i>0) and (i<sizeX-1) and (j>0) and (j<sizeY-1):
                print("u["+str(i+1)+str(j)+"] - u["+str(i-1)+str(j)+"] + " + "v["+str(i)+str(j+1)+"] - u["+str(i)+str(j-1)+"] "  )
                s+= u[indexTo1D(i+1,j, sizeX),0].numpy() - u[indexTo1D(i-1,j, sizeX),0].numpy() 
                s+= v[indexTo1D(i,j+1, sizeX),0].numpy()  - v[indexTo1D(i,j-1, sizeX),0].numpy() 
            div.append(s)

    div = tf.convert_to_tensor(div, dtype=tf.float32)
    if tf.rank(div).numpy() < 2:
        div = tf.expand_dims(div, 1)
    return div

def compute_divergence2(u, v, sizeX, sizeY, h):
    dx = tf.roll(u, shift=-1, axis=0) - tf.roll(u, shift=1, axis=0)
    dy = tf.roll(v, shift=-sizeX, axis=0) - tf.roll(v, shift=sizeX, axis=0)
    div = (dx + dy)
    if tf.rank(div).numpy() < 2:
        div = tf.expand_dims(div, 1)
    return div

def solvePressure(u, v, sizeX, sizeY, h, mat):
    dx = tf.roll(u, shift=-1, axis=0) - tf.roll(u, shift=1, axis=0)
    dy = tf.roll(v, shift=-sizeX, axis=0) - tf.roll(v, shift=sizeX, axis=0)
    div = (dx + dy)*0.5/h
    if tf.rank(div).numpy() < 2:
        div = tf.expand_dims(div, 1)
    return tf.linalg.solve(mat, div)

def project1(u,v, sizeX, sizeY, mat, h, boundary_func):
    _u, _v = set_solid_boundary(u,v, sizeX, sizeY, boundary_func)
    p = solvePressure(_u,_v,sizeX,sizeY,h, mat)[..., 0]
    gradP_u = []
    gradP_v = []
    for j in range(sizeY):
        for i in range(sizeX):
            if (i>0) and (i < sizeX-1):
                gradP_u.append(p[indexTo1D(i+1,j,sizeX)] - p [indexTo1D(i-1, j, sizeX)])
            else:
                gradP_u.append(0)
            if (j>0) and (j < sizeY-1):
                gradP_v.append(p[indexTo1D(i,j+1,sizeX)] - p[indexTo1D(i,j-1,sizeX)])
            else:
                gradP_v.append(0)
    gradP_u = (0.5/h)*tf.convert_to_tensor(gradP_u)
    gradP_v = (0.5/h)*tf.convert_to_tensor(gradP_v)
            
    if tf.rank(gradP_u) < tf.rank(u):
        gradP_u = tf.expand_dims(gradP_u, 1)
        gradP_v = tf.expand_dims(gradP_v, 1)

    new_u = _u - gradP_u
    new_v = _v - gradP_v
    return set_solid_boundary(new_u, new_v, sizeX, sizeY,boundary_func)

def project2(u,v, sizeX, sizeY, mat, h, boundary_func):
    _u, _v = set_solid_boundary(u,v, sizeX, sizeY, boundary_func)
    p = solvePressure(_u,_v,sizeX,sizeY,h, mat)[..., 0]
    gradP_u = (0.5/h)*(tf.roll(p, shift=-1, axis=0) - tf.roll(p, shift=1, axis=0))
    gradP_v = (0.5/h)*(tf.roll(p, shift=-sizeX, axis=0) - tf.roll(p, shift=sizeX, axis=0))
    if tf.rank(gradP_u) < tf.rank(u):
        gradP_u = tf.expand_dims(gradP_u, 1)
        gradP_v = tf.expand_dims(gradP_v, 1)
    new_u = _u - gradP_u
    new_v = _v - gradP_v
    return set_solid_boundary(new_u, new_v, sizeX, sizeY,boundary_func)

@tf.function
def create_vortex(center, r, w, coords, alpha=1.0):
    rel_coords = coords - center
    dist = tf.linalg.norm(rel_coords, axis=-1)
    smoothed_dist = tf.exp(-tf.pow(dist*alpha/r,2.0))
    u = w*rel_coords[...,1] * smoothed_dist
    v = - w*rel_coords[..., 0] * smoothed_dist
    return u,v

@tf.function
def init_vortices(n, centers, radius, w, coords, size):
    u = tf.zeros([size*size])
    v = tf.zeros([size*size])
    for i in range(n):
        u_tmp,v_tmp = create_vortex(centers[i], radius[i], w[i], coords) 
        u += u_tmp
        v += v_tmp
    return u,v


if len(sys.argv) > 1:
    size = 4

    if sys.argv[1] == "masking":
        # Create a tensor of shape (m)
        tensor = tf.range(size*size)

        # Define the value of n
        n = size

        # Compute a mask where every element whose index is 0 % n is True
        mask = tf.logical_or(tf.logical_or(tf.math.equal(tf.range(tf.shape(tensor)[0]) % n, 0), tf.math.equal(tf.range(tf.shape(tensor)[0]) % n, n-1)),
                            tf.logical_or(tf.math.equal(tf.range(tf.shape(tensor)[0]) // n, 0), tf.math.equal(tf.range(tf.shape(tensor)[0]) // n, n-1)))

        print(tensorToGrid(mask, size, size))  # [1 4 7 10]

    if sys.argv[1] == "matrix":
        size = 25
        # Generate indices, values, and dense_shape for the sparse tensor
        indices = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]  # Indices of non-zero elements
        values = [1, 2, 3, 4, 5]  # Values of non-zero elements
        dense_shape = [size*size, size*size]  # Shape of the dense tensor
        
        print("Create Sparse Matrix")
        start_time = time.time()
        sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
        end_time = time.time()
        print("Execution time: {:.4f} seconds".format(end_time-start_time))


        a = tf.linalg.lu(sparse_tensor)

        # print("Building Laplacian Matrix...")
        # mat = build_laplacian_matrix(25, 25, 1.0, -4.0)
        # print(np.shape(mat[0]))
        # print(mat[0])
        # tf_mat = tf_build_laplacian_matrix(size, size, 1.0, -4.0)
        # print(tf_mat[0])

    if sys.argv[1] == "div":
        x = tf.constant([[1, 5, 9, 13],
                        [2, 6, 10, 14],
                        [3, 7, 11, 15],
                        [4, 8, 12, 16]])
        _u = tf.random.normal((size*size,))
        _v = tf.random.normal((size*size,))
        u,v = set_solid_boundary(_u,  _v, size, size) 

        h = 0.01

        div1 = compute_divergence1(u, v, size, size, h)
        div2 = compute_divergence2(u, v, size, size, h)
        # div1, div2 = set_solid_boundary(div1, div2, size, size) 
        print("div 1 = ", div1)
        print("div 2 = ", div2)

    if sys.argv[1] == "project":
        h = 2/size
        _u = tf.random.normal((size*size,1))
        _v = tf.random.normal((size*size,1))
        u,v = set_solid_boundary(_u,  _v, size, size)
        mat = tf.convert_to_tensor(build_laplacian_matrix(size, size, 1/(h*h), -4/(h*h)), dtype=tf.float32)

        u1, v1 = project1(u, v, size, size, mat, h, 0)
        u2, v2 = project2(u, v, size, size, mat, h, 0)
        print(u1)
        print(u2)

    if sys.argv[1] == "data":
        print("Python v", sys.version)
        print("TensorFlow v", tf.__version__)

    if sys.argv[1] == "constraint":
        CONSTRAINT = {}
        CONSTRAINT_FILE = "test_density.json"
        with open("../data/"+CONSTRAINT_FILE) as file:
            print('Loading file', CONSTRAINT_FILE)
            CONSTRAINT = json.load(file)

        target_density = CONSTRAINT["target_density"]
        init_density = CONSTRAINT["init_density"]
        SIZE = int(np.sqrt(len(target_density)))

        FLUID_SETTINGS = {}
        FLUID_SETTINGS["timestep"] = 0.025
        FLUID_SETTINGS["grid_min"] = -1
        FLUID_SETTINGS["grid_max"] = 1
        FLUID_SETTINGS["diffusion_coeff"] = 0.0
        FLUID_SETTINGS["dissipation_rate"] = 0.0
        FLUID_SETTINGS["viscosity"] = 0.0
        FLUID_SETTINGS["source"] = None
        D = (FLUID_SETTINGS["grid_max"] -FLUID_SETTINGS["grid_min"])/SIZE

        COORDS_X = []   # x-coordinates of position
        COORDS_Y = []   # y-coordinates of position
        for j in range(SIZE):
            for i in range(SIZE):
                point_x = FLUID_SETTINGS["grid_min"]+(i+0.5)*D
                point_y = FLUID_SETTINGS["grid_min"]+(j+0.5)*D
                COORDS_X.append(point_x)
                COORDS_Y.append(point_y)


        indices = np.array(CONSTRAINT["indices"])
        u = np.zeros(SIZE*SIZE)
        v = np.zeros(SIZE*SIZE)
        for i, idx  in enumerate(indices):
            u[idx] = CONSTRAINT["values"][i][0][0]*D
            v[idx] = CONSTRAINT["values"][i][1][0]*D

        x,y = np.meshgrid(COORDS_X[:SIZE], COORDS_Y[::SIZE])
        u = np.reshape(u, (SIZE,SIZE))
        v = np.reshape(v, (SIZE,SIZE))
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        Q = ax.quiver(x, y, u, v, color='red', scale_units='width')

        density_t = erf(np.clip(np.flipud(np.reshape(target_density, (SIZE,SIZE))), 0, None) * 2)
        density_i = erf(np.clip(np.flipud(np.reshape(init_density, (SIZE,SIZE))), 0, None) * 2)
        img_t = Image.fromarray((cm.cividis(density_t) * 255).astype('uint8')).resize((640, 640))
        img_i = Image.fromarray((cm.cividis(density_i) * 255).astype('uint8')).resize((640, 640))
        img_t.show()
        img_i.show()
        plt.show()


    if sys.argv[1] == "vortices":
        # Calculate some useful physicial quantities
        GRID_MAX = 1
        GRID_MIN = -1
        SIZE = 25
        D = (GRID_MAX - GRID_MIN)/SIZE
        COORDS_X = []   # x-coordinates of position
        COORDS_Y = []   # y-coordinates of position
        NB_VORTICES = 4

        for j in range(SIZE):
            for i in range(SIZE):
                point_x = GRID_MIN+(i+0.5)*D
                point_y = GRID_MIN+(j+0.5)*D
                COORDS_X.append(point_x)
                COORDS_Y.append(point_y)

        COORDS = tf.stack((COORDS_X, COORDS_Y), axis=1)
        centers_init = tf.convert_to_tensor([[0.5, -0.6], [-0.3, -0.5], [0.1, 0], [-0.4, 0.4] ]  , dtype=tf.float32)

        centers = tf.Variable(tf.random.uniform([NB_VORTICES,2], minval=GRID_MIN, maxval=GRID_MAX))
        centers = tf.Variable(centers_init)
        # radius = tf.Variable(tf.random.uniform([NB_VORTICES]))
        radius = tf.Variable(0.4*tf.convert_to_tensor([1, 1, 1, 1]  , dtype=tf.float32)) 
        # w = tf.Variable(tf.random.normal([NB_VORTICES]))
        w = tf.Variable(0.1*tf.convert_to_tensor([1, 1, -2, 2]  , dtype=tf.float32)) 

        
        # Gradient
        with tf.GradientTape() as tape:
            u,v = init_vortices(NB_VORTICES, centers, radius, w, COORDS, SIZE)
        grad = tape.gradient([u,v], [centers, radius, w])
        
        # Differentiability test
        if all([gradient is not None for gradient in grad]):
            print(colored("init_vortex is differentiable.", 'green'))
        else:
            print(colored("init_vortex is not differentiable.", 'red'))
        print(grad)

        # Affichage
        x,y = np.meshgrid(COORDS_X[:SIZE], COORDS_Y[::SIZE])
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        u_viz = tf.reshape(u, shape=(SIZE, SIZE)).numpy()
        v_viz = tf.reshape(v, shape=(SIZE, SIZE)).numpy()
        Q = ax.quiver(x, y, u, v, color='red', scale_units='width')
        plt.show()
    
    if sys.argv[1] == "loss":
        print("Loss")
        size = 2
        a = tf.range(size, dtype=tf.float32)
        b = 2*tf.range(size, dtype=tf.float32)
        c = 3*tf.range(size, dtype=tf.float32)
        d = tf.random.uniform([size], dtype=tf.float32)
        midVel = [1+a,1+b]
        # w = [i+1  for i in range(len(midVel))]
        # print("a = ", a)
        # print("midVel = ",midVel)
        # loss = tf.constant(0, dtype=tf.float32)
        # for t in range(len(midVel)):
        #     loss += w[t]* tf.norm(tf.reshape(midVel[t],  [-1]))
        #     print(w[t]*tf.reshape(midVel[t],  [-1]))
        # print("loss = ", loss)


        values =  [[[-2, 2], [0, 0]],[[-1, 0], [0, 0]]]
        values_pred =  [[[2, -2], [0, 0]],[[-1, 0], [0, 0]]]
        depth = 2*len(values[0][0])
        w = tf.convert_to_tensor([i+1  for i in range(len(values))], dtype=tf.float32)

        y = tf.reshape(tf.convert_to_tensor(values, dtype=tf.float32), (-1, depth))
        y_pred = tf.reshape(tf.convert_to_tensor(values_pred, dtype=tf.float32), (-1, depth))

        print("y = ", y)
        loss =  tf.multiply(w,1.0+tf.keras.losses.cosine_similarity(y, y_pred))
        print("Cosine loss = ", loss)
        print(" Loss = ", tf.reduce_sum(loss))

        # a = tf.concat(midVel, 0)
        # print("a = ", a)



