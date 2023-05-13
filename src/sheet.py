import tensorflow as tf
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import erf
import matplotlib.cm as cm
from termcolor import colored


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
    mat = tf.zeros((sizeX*sizeY, sizeX*sizeY))
    indices = tf.range(sizeX*sizeY, dtype=tf.int32)

    i = indices % sizeX
    j = indices // sizeX

    mat = tf.linalg.set_diag(mat, tf.fill((sizeX*sizeY,), b))
    # print("mat = ", mat)

    if a != 0:
        mask_left = tf.math.greater(i, 1)
        print("mask_left = ",  mask_left)
        left_idx = tf.boolean_mask(indices, mask_left)
        mat = tf.tensor_scatter_nd_update(mat, tf.expand_dims(left_idx - 1, axis=1), tf.reshape(tf.fill(tf.shape(left_idx), a), (-1, 1)))

        # mask_right = tf.math.less(i, sizeX - 1)
        # right_idx = tf.boolean_mask(indices, mask_right)
        # mat = tf.tensor_scatter_nd_update(mat, tf.expand_dims(right_idx + 1, axis=1), tf.fill(tf.shape(right_idx), a))

        # mask_top = tf.math.greater(j, 1)
        # top_idx = tf.boolean_mask(indices, mask_top)
        # mat = tf.tensor_scatter_nd_update(mat, tf.expand_dims(top_idx - sizeX, axis=1), tf.fill(tf.shape(top_idx), a))

        # mask_bottom = tf.math.less(j, sizeY - 1)
        # bottom_idx = tf.boolean_mask(indices, mask_bottom)
        # mat = tf.tensor_scatter_nd_update(mat, tf.expand_dims(bottom_idx + sizeX, axis=1), tf.fill(tf.shape(bottom_idx), a))

    return mat

def build_laplacian_matrix(sizeX,sizeY,a,b):
    mat = np.zeros((sizeX*sizeY,sizeX*sizeY))
    for it in range(sizeX*sizeY):
        i = it%sizeX
        j = it//sizeX
        mat[it,it] = b
        if (i>1):
            mat[it,it-1] = a
        if (i<sizeX-1):
            mat[it, it+1] = a
        if (j>1):
            mat[it,it-sizeX] = a
        if (j<sizeY-1):
            mat[it, it+sizeX] = a       
    return mat

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
    smoothed_dist = tf.exp(-tf.pow((dist-r)*alpha/r,2.0))
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
        mat = build_laplacian_matrix(size, size, 1.0, -4.0)
        print(mat)
        tf_mat = tf_build_laplacian_matrix(size, size, 1.0, -4.0)
        print(tf_mat)

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
        NB_VORTICES = 3

        for j in range(SIZE):
            for i in range(SIZE):
                point_x = GRID_MIN+(i+0.5)*D
                point_y = GRID_MIN+(j+0.5)*D
                COORDS_X.append(point_x)
                COORDS_Y.append(point_y)

        COORDS = tf.stack((COORDS_X, COORDS_Y), axis=1)
        centers = tf.Variable(tf.random.uniform([NB_VORTICES,2], minval=-1))
        radius = tf.Variable(tf.random.uniform([NB_VORTICES]))
        w = tf.Variable(tf.random.normal([NB_VORTICES]))
        
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
         

