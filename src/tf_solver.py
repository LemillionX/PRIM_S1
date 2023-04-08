import tensorflow as tf
import numpy as np
import sys
from termcolor import colored
from tqdm import tqdm


@tf.function
def addSource(s, value=1.0, indices=None):
    return tf.tensor_scatter_nd_update(s, indices, tf.constant(value, shape=[indices.shape[0]], dtype=tf.float32))

@tf.function
def set_absorbing_boundary(u, v, sizeX, sizeY, b=0):
    new_u = tf.identity(u)
    new_v = tf.identity(v)
    mask = tf.logical_or(tf.logical_or(tf.math.equal(tf.range(sizeX*sizeY) % sizeX, 0), tf.math.equal(tf.range(sizeX*sizeY) % sizeX, sizeX-1)),
                     tf.logical_or(tf.math.equal(tf.range(sizeX*sizeY) // sizeX, 0), tf.math.equal(tf.range(sizeX*sizeY) // sizeX, sizeY-1)))
    indices = tf.where(mask)
    indices.set_shape((2*(sizeX + sizeY-2), 1))
    new_u = tf.tensor_scatter_nd_update(new_u, indices, tf.constant(b, shape=[indices.shape[0]], dtype=tf.float32))
    new_v = tf.tensor_scatter_nd_update(new_v, indices, tf.constant(b, shape=[indices.shape[0]], dtype=tf.float32))
    return new_u, new_v

@tf.function
def set_solid_boundary(u,v,sizeX,sizeY,b=0):
    new_u = tf.identity(u)
    new_v = tf.identity(v)
    mask_down = tf.expand_dims(tf.range(0, sizeX, 1),1)
    mask_up = tf.expand_dims(tf.range(sizeX*(sizeY-1), sizeX*sizeY, 1), 1)
    mask_left = tf.expand_dims(tf.range(0, sizeX*sizeY, sizeX),1)
    mask_right = tf.expand_dims(tf.range(sizeX-1,sizeX*sizeY,sizeX), 1)

    # Left boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_left, tf.abs(tf.gather_nd(tf.roll(u, shift=-1, axis=0), mask_left)))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_left, tf.gather_nd(tf.roll(v, shift=-1, axis=0), mask_left))
    # Right boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_right, -tf.abs(tf.gather_nd(tf.roll(u, shift=1, axis=0), mask_right)))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_right, tf.gather_nd(tf.roll(v, shift=1, axis=0), mask_right))
    # Up boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_up, tf.gather_nd(tf.roll(u, shift=sizeX, axis=0), mask_up))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_up, -tf.abs(tf.gather_nd(tf.roll(v, shift=sizeX, axis=0), mask_up)))
    # Down boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_down, tf.gather_nd(tf.roll(u, shift=-sizeX, axis=0), mask_down))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_down, tf.abs(tf.gather_nd(tf.roll(v, shift=-sizeX, axis=0), mask_down)))
    
    # Upper-left corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[indexTo1D(0, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_u[indexTo1D(0, sizeY-2, sizeX)] + new_u[indexTo1D(1, sizeY-1, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[indexTo1D(0, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_v[indexTo1D(0, sizeY-2, sizeX)] + new_v[indexTo1D(1, sizeY-1, sizeX)]),0))
    # Upper-right corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[indexTo1D(sizeX-1, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_u[indexTo1D(sizeX-1, sizeY-2, sizeX)] + new_u[indexTo1D(sizeX-2, sizeY-1, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[indexTo1D(sizeX-1, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_v[indexTo1D(sizeX-1, sizeY-2, sizeX)] + new_v[indexTo1D(sizeX-2, sizeY-1, sizeX)]),0))
    # Bottom-left corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[indexTo1D(0, 0, sizeX)]], 0.5*tf.expand_dims((new_u[indexTo1D(0, 1, sizeX)] + new_u[indexTo1D(1, 0, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[indexTo1D(0, 0, sizeX)]], 0.5*tf.expand_dims((new_v[indexTo1D(0, 1, sizeX)] + new_v[indexTo1D(1, 0, sizeX)]),0))
    # Bottom-right corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[indexTo1D(sizeX-1, 0, sizeX)]], 0.5*tf.expand_dims((new_u[indexTo1D(sizeX-2, 0, sizeX)] + new_u[indexTo1D(sizeX-1, 1, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[indexTo1D(sizeX-1, 0, sizeX)]], 0.5*tf.expand_dims((new_v[indexTo1D(sizeX-2, 0, sizeX)] + new_v[indexTo1D(sizeX-1, 1, sizeX)]),0))


    return new_u, new_v


@tf.function
def set_scalar_boundary(s, sizeX, sizeY):
    new_s = tf.identity(s)
    mask_down = tf.expand_dims(tf.range(0, sizeX, 1),1)
    mask_up = tf.expand_dims(tf.range(sizeX*(sizeY-1), sizeX*sizeY, 1), 1)
    mask_left = tf.expand_dims(tf.range(0, sizeX*sizeY, sizeX),1)
    mask_right = tf.expand_dims(tf.range(sizeX-1,sizeX*sizeY,sizeX), 1)
    new_s = tf.tensor_scatter_nd_update(new_s, mask_left, tf.gather_nd(tf.roll(s, shift=-1, axis=0), mask_left))
    new_s = tf.tensor_scatter_nd_update(new_s, mask_right, tf.gather_nd(tf.roll(s, shift=1, axis=0), mask_right))
    new_s = tf.tensor_scatter_nd_update(new_s, mask_up, tf.gather_nd(tf.roll(s, shift=sizeX, axis=0), mask_up))
    new_s = tf.tensor_scatter_nd_update(new_s, mask_down, tf.gather_nd(tf.roll(s, shift=-sizeX, axis=0), mask_down))
    # Upper-left corner
    new_s = tf.tensor_scatter_nd_update(new_s, [[indexTo1D(0, sizeY-1, sizeX)]], 0.5*tf.expand_dims((s[indexTo1D(0, sizeY-2, sizeX)] + s[indexTo1D(1, sizeY-1, sizeX)]),0))
    # Upper-right corner
    new_s = tf.tensor_scatter_nd_update(new_s, [[indexTo1D(sizeX-1, sizeY-1, sizeX)]], 0.5*tf.expand_dims((s[indexTo1D(sizeX-1, sizeY-2, sizeX)] + s[indexTo1D(sizeX-2, sizeY-1, sizeX)]),0))
    # Bottom-left corner
    new_s = tf.tensor_scatter_nd_update(new_s, [[indexTo1D(0, 0, sizeX)]], 0.5*tf.expand_dims((s[indexTo1D(0, 1, sizeX)] + s[indexTo1D(1, 0, sizeX)]),0))
    # Bottom-right corner
    new_s = tf.tensor_scatter_nd_update(new_s, [[indexTo1D(sizeX-1, 0, sizeX)]], 0.5*tf.expand_dims((s[indexTo1D(sizeX-2, 0, sizeX)] + s[indexTo1D(sizeX-1, 1, sizeX)]),0))
    return new_s

@tf.function
def indexTo1D(i,j, sizeX):
    return j*sizeX+i

@tf.function
def set_boundary(u,v,sizeX,sizeY,boundary_func=None,b=0):
    if boundary_func is None:
        return set_solid_boundary(u,v,sizeX,sizeY,b)
    else:
        return boundary_func(u,v,sizeX,sizeY,b)

@tf.function
def sampleAt(x,y, data, sizeX, sizeY, offset, d):
    _x = (x - offset)/d - 0.5
    _y = (y - offset)/d - 0.5
    i0 = tf.clip_by_value(tf.floor(_x), 0, sizeX-1)
    j0 = tf.clip_by_value(tf.floor(_y), 0, sizeY-1)
    i1 = tf.clip_by_value(i0+1, 0, sizeX-1)
    j1 = tf.clip_by_value(j0+1, 0, sizeY-1)

    p00 = tf.gather(data, tf.cast(indexTo1D(i0,j0,sizeX), tf.int32))
    p01 = tf.gather(data, tf.cast(indexTo1D(i0,j1,sizeX), tf.int32))
    p10 = tf.gather(data, tf.cast(indexTo1D(i1,j0,sizeX), tf.int32))
    p11 = tf.gather(data, tf.cast(indexTo1D(i1,j1,sizeX), tf.int32))

    t_i0 = (offset + (i0+1+0.5)*d -x)/d
    t_j0 = (offset + (j0+1+0.5)*d -y)/d
    t_i1 = (x - (offset + (i0+0.5)*d))/d
    t_j1 = (y - (offset + (j0+0.5)*d))/d

    return t_i0*t_j0*p00 + t_i0*t_j1*p01 + t_i1*t_j0*p10 + t_i1*t_j1*p11

@tf.function
def advectCentered(f, u,v, sizeX, sizeY, coords_x, coords_y, dt, offset, d):
    traced_x = tf.clip_by_value(coords_x - dt*u, offset + 0.5*d, offset + (sizeX-0.5)*d )
    traced_y = tf.clip_by_value(coords_y - dt*v, offset + 0.5*d, offset + (sizeY-0.5)*d)
    u1 = tf.vectorized_map(fn=lambda x: sampleAt(x[0],x[1],f,sizeX,sizeY, offset, d), elems=(traced_x, traced_y))
    # return set_boundary(u1, sizeX, sizeY)
    return u1

def build_laplacian_matrix(sizeX,sizeY,a,b):
    mat = np.zeros((sizeX*sizeY,sizeX*sizeY))
    for it in range(sizeX*sizeY):
        i = it%sizeX
        j = it//sizeX
        mat[it,it] = b
        if (i>0):
            mat[it,it-1] = a
        if (i<sizeX-1):
            mat[it, it+1] = a
        if (j>0):
            mat[it,it-sizeX] = a
        if (j<sizeY-1):
            mat[it, it+sizeX] = a
    return mat

def diffuse(f, mat):
    if tf.rank(f).numpy() < 2:
        f = tf.expand_dims(f, 1)
    return tf.linalg.solve(mat,f)

def solvePressure(u, v, sizeX, sizeY, h, mat):
    dx = tf.roll(u, shift=-1, axis=0) - tf.roll(u, shift=1, axis=0)
    dy = tf.roll(v, shift=-sizeX, axis=0) - tf.roll(v, shift=sizeX, axis=0)
    div = (dx + dy)*0.5/h
    div = set_scalar_boundary(div, sizeX, sizeY)
    if tf.rank(div) < 2:
        div = tf.expand_dims(div, 1)
    return tf.linalg.solve(mat, div)

def project(u,v, sizeX, sizeY, mat, h, boundary_func):
    _u, _v = set_boundary(u,v, sizeX, sizeY, boundary_func)
    p = solvePressure(_u,_v,sizeX,sizeY,h, mat)[..., 0]
    p = set_scalar_boundary(p, sizeX, sizeY)

    gradP_u = (0.5/h)*(tf.roll(p, shift=-1, axis=0) - tf.roll(p, shift=1, axis=0))
    gradP_v = (0.5/h)*(tf.roll(p, shift=-sizeX, axis=0) - tf.roll(p, shift=sizeX, axis=0))
            
    new_u = _u - gradP_u
    new_v = _v - gradP_v
    return set_boundary(new_u, new_v, sizeX, sizeY,boundary_func)

def dissipate(s,a,dt):
    return s/(1+dt*a)

def update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, boundary_func=None, source=None, t=np.inf):
    ## Vstep
    # advection step
    new_u = advectCentered(_u, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    new_v = advectCentered(_v, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    _u = new_u
    _v = new_v

    # diffusion step
    if _visc > 0:
        _u = diffuse(_u, _vDiff_mat)[..., 0]
        _v = diffuse(_v, _vDiff_mat)[..., 0]
        _u, _v = set_boundary(_u, _v, _sizeX, _sizeY, boundary_func) 
    # projection step
    _u, _v = project(_u, _v, _sizeX, _sizeY, _mat, _h, boundary_func)


    ## Sstep
    if (source is not None) and (t < source["time"]) :
        _s = addSource(_s, source["value"], source["indices"])
    # advection step
    _s = advectCentered(_s, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)

    # diffusion step
    if _kDiff > 0:
        _s = diffuse(_s, _sDiff_mat)
        _s = set_scalar_boundary(_s, _sizeX, _sizeY)

    # dissipation step
    if _alpha > 0:
        _s = dissipate(_s, _alpha, _dt)
    
    if (source is not None) and (t < source["time"]) :
        _s = addSource(_s, source["value"], source["indices"])
    return _u, _v, _s

def simulate(n_iter, _u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, boundary_func=None, source=None, leave=True):
    for t in tqdm(range(1, n_iter+1), desc = "Simulating....", leave=leave):
        new_u, new_v, new_s = update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff,boundary_func, source, t)
        _u = new_u
        _v = new_v
        _s = new_s
    return new_u, new_v, new_s

def simulateConstrained(n_iter, _u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, keyframes=[], keyidx=[], boundary_func=None, source=None, leave=True):
    _midVel = []
    count = -1
    for t in tqdm(range(1, n_iter+1), desc = "Simulating....", leave=leave):
        new_u, new_v, new_s = update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff,boundary_func, source, t)
        _u = new_u
        _v = new_v
        _s = new_s
        if t in keyframes:
            count += 1
            _midVel.append([tf.identity(_u[keyidx[count]]), tf.identity(_v[keyidx[count]])])
    return new_u, new_v, new_s, _midVel

if __name__ in ["__main__", "__builtin__"]:
    _sizeX = 5
    _sizeY = 5
    _coordsX = []
    _coordsY = []
    _grid_min = -1
    _grid_max = 1
    _d = (_grid_max - _grid_min)/_sizeX

    for j in range(_sizeY):
        for i in range(_sizeX):
            _coordsX.append(_grid_min + (i+0.5)*_d)
            _coordsY.append(_grid_min + (j+0.5)*_d)

    _coordsX = tf.convert_to_tensor(_coordsX, dtype=tf.float32)
    _coordsY = tf.convert_to_tensor(_coordsY, dtype=tf.float32)

    if len(sys.argv) > 1:
        if sys.argv[1] == "index":
            # Variables
            i0 = tf.Variable(10*tf.random.normal([1]))
            j0 = tf.Variable(10*tf.random.normal([1]))

            # Gradient
            with tf.GradientTape() as tape:
                idx = indexTo1D(i0,j0,_sizeX)
            grad_index = tape.gradient(idx, [i0, j0])

            # Differentiability test
            if all([gradient is not None for gradient in grad_index]):
                print(colored("indexTo1D is differentiable.", 'green'))
            else:
                print(colored("indexTo1D is not differentiable.", 'red'))

        if sys.argv[1] == "sample":
            # Variables
            _pos = tf.Variable(tf.random.normal([2]))
            _u = tf.Variable(tf.random.normal([_sizeX*_sizeY]))

            # Gradient
            with tf.GradientTape() as tape:
                _value = sampleAt(_pos[0], _pos[1], _u, _sizeX, _sizeY, _grid_min, _d)
            grad_sampleAt = tape.gradient([_value], [_pos, _u])

            # Differentiability test
            if all([gradient is not None for gradient in grad_sampleAt]):
                print(colored("sampleAt is differentiable.", 'green'))
            else:
                print(colored("sampleAt is not differentiable.", 'red'))

        if sys.argv[1] == "advect":
            print("*******************Testing advection*********************")
            # Variables
            _u = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _v = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _dt = tf.Variable(tf.random.normal([1])) 

            # Gradients
            with tf.GradientTape() as tape:
                new_u = advectCentered(_u, _u, _v, _sizeX, _sizeY, _coordsX, _coordsY, _dt, _grid_min, _d)
            print("Executing OK")
            grad_advection = tape.gradient([new_u], [ _u, _v, _dt])

            # Differentiability test
            if all([gradient is not None for gradient in grad_advection]):
                print(colored("The advection solver is differentiable.", 'green'))
            else:
                print(colored("The advection solver is not differentiable.", 'red'))

        if sys.argv[1] == "diffuse":
            print("********************Testing diffusion***********************")

            # Variables
            _u = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _visc = tf.Variable(tf.random.normal([1]))

            # Gradients
            with tf.GradientTape() as tape:
                _diffuse_mat = tf.convert_to_tensor(build_laplacian_matrix(_sizeX, _sizeY, -_visc/(_d*_d), 1+4*_visc/(_d*_d) ), dtype=tf.float32)
                new_u = diffuse(_u, _diffuse_mat)
            print("Executing OK")
            grad_diff = tape.gradient([new_u], [ _u])

            # Differentiability test
            if all([gradient is not None for gradient in grad_diff]):
                print(colored("The advection solver is differentiable.", 'green'))
            else:
                print(colored("The advection solver is not differentiable.", 'red'))

        if sys.argv[1] == "project":
            print("*******************Testing projection*********************")
            # Variables
            _u = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _v = tf.Variable(tf.random.normal([_sizeX*_sizeY]))

            # Gradients
            with tf.GradientTape() as tape:
                _mat = tf.convert_to_tensor(build_laplacian_matrix(_sizeX, _sizeY,1,-4), dtype=tf.float32)
                new_u, new_v = project(_u, _v, _sizeX, _sizeY, _mat, _d)
            print("Executing OK")
            grad_projectU = tape.gradient([new_u, new_v], [ _u, _v])

            # Differentiability test
            if all([gradient is not None for gradient in grad_projectU]):
                print(colored("The projection solver is differentiable.", 'green'))
            else:
                print(colored("The projection solver is not differentiable.", 'red'))            

        if sys.argv[1] == "dissipate":
            print("************************Testing dissipation************************")
            # Variables
            _s = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _kDiff = tf.Variable(tf.random.normal([1]))
            _dt = tf.Variable(tf.random.normal([1]))

            # Gradients
            with tf.GradientTape() as tape:
                _u = dissipate(_s, _kDiff, _dt)
            print("Executing OK")
            grad_dissipation = tape.gradient([_u], [_s, _kDiff, _dt])

            # Differentiability test
            if all([gradient is not None for gradient in grad_dissipation]):
                print(colored("The dissipation solver is differentiable.", 'green'))
            else:
                print(colored("The dissipation solver is not differentiable.", 'red'))   

        if sys.argv[1] == "gradient":
            print("**************Testing solver****************")          
            
            # Variables
            _u = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _v = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _s = tf.Variable(tf.random.normal([_sizeX*_sizeY]))
            _dt = tf.Variable(tf.constant(0.025))
            _visc = tf.random.normal([1])
            _kDiff = tf.random.normal([1])
            _alpha = tf.random.normal([1])
            
            # Gradients
            with tf.GradientTape() as tape:
                _Sdiffuse_mat = tf.convert_to_tensor(build_laplacian_matrix(_sizeX, _sizeY, -_kDiff/(_d*_d), 1+4*_kDiff/(_d*_d) ), dtype=tf.float32)
                _Vdiffuse_mat = tf.convert_to_tensor(build_laplacian_matrix(_sizeX, _sizeY, -_visc/(_d*_d), 1+4*_visc/(_d*_d) ), dtype=tf.float32)
                _laplacian_mat = tf.convert_to_tensor(build_laplacian_matrix(_sizeX, _sizeY,1/(_d*_d),-4/(_d*_d)), dtype=tf.float32)
                new_u, new_v, new_s = update(_u, _v, _s, _sizeX, _sizeY, _coordsX, _coordsY, _dt, _grid_min, _d, _laplacian_mat, _alpha, _Vdiffuse_mat, _visc, _Sdiffuse_mat, _kDiff)
            print("Executing OK")
            grad_solver = tape.gradient([new_s], [_u, _v, _s, _dt])

            # print(grad_solver)

            # Differentiability test
            if all([gradient is not None for gradient in grad_solver]):
                print(colored("The solver is differentiable.", 'green'))
            else:
                print(colored("The solver is not differentiable.", 'red'))











