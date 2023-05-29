'''
A TensorFlow version of a Stable Fluid solver, using Staggered Grid  

:author:    Sammy Rasamimanana
:year:      2023
'''
import tensorflow as tf
import numpy as np
import sys
from termcolor import colored
from tqdm import tqdm

@tf.function
def curl2Dvector(a, sizeX, h):
    '''
    Return the 2D curl vector (da/dy, -da/dx) of a 2D scalar field ``a``.

    Args:
        a: A TensorFlow ``tensor`` of shape ``(sizeX, sizeX)`` representing a 2D scalar field in a grid of size ``(sizeX, sizeX)``
        sizeX: An ``int`` representing the length and the width of the grid
    
    Returns:
        dy: A TensorFlow ``tensor`` of shape ``(sizeX, sizeX)`` representing da/dy
        -dx: A TensorFlow ``tensor`` of shape ``(sizeX, sizeX)`` representing -da/dx
    '''
    dx = tf.roll(a, shift=-1, axis=0) - tf.roll(a, shift=1, axis=0)
    dy = tf.roll(a, shift=-sizeX, axis=0) - tf.roll(a, shift=sizeX, axis=0)
    return 0.5*dy/h, -0.5*dx/h

@tf.function
def addSource(s, value=1.0, indices=None):
    '''
    Return the updated density grid ``s`` such that ``s[indices]`` = ``value``

    Args:
        s: A TensorFlow ``tensor`` representing the density grid
        value: A ``float`` representing the value to put at ``indices``. Default is set to ``1.0``
        indices: A TensorFlow ``tensor`` of shape (N, 1, 1) where N is the number of cells to update in the grid ``s``. Default is set to ``None``
    '''
    return tf.tensor_scatter_nd_update(s, indices, tf.constant(value, shape=[indices.shape[0]], dtype=tf.float32))

@tf.function
def set_solid_boundary(u, v, sizeX, sizeY, b=0):
    '''
    Set the boundaries of the grids ``u`` and ``v`` to the value ``b``

    Args:
        u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        sizeX: An ``int`` representing the number of horizontal cells
        sizeY: An ``int`` representing the number of vertical cells
        b: A ``float`` representing the value at the boundaries. Default is set to ``0``

    Returns:
        new_u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the updated version of ``u``
        new_v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the updated version of ``v``
    '''
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
def indexTo1D(i,j, sizeX):
    '''
    Gives the 1D index of the 2D index ``[i,j]`` for a 1D grid of size ``sizeX*sizeX`` representing a 2D Grid of shape ``(sizeX, sizeX)``

    Args:
        i: An int 
        j: An int 
        sizeX: An int

    Returns:
        ``i+j*sizeX``: An int
    '''
    return j*sizeX+i

@tf.function
def set_boundary(u,v,sizeX,sizeY,boundary_func=None,b=0):
    '''
    Applies the boundary function ``boundary_func`` to ``u`` and ``v``.
    
    Args:
        u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        sizeX: An ``int`` representing the number of horizontal cells
        sizeY: An ``int`` representing the number of vertical cells
        boundary_func: A function of signature (``tensor``, ``tensor``, ``int``, ``int``, ``float``) -> (``tensor``, ``tensor``) where the tensors must have the same shape. Default is set to ``None``, in that case the solid boundaries are applied.
        b: A ``float``  set to ``0`` to ensure compatibility with customized boundary functions.

    Returns:
        new_u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the updated version of ``u``
        new_v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the updated version of ``v``
    '''
    if boundary_func is None:
        return set_solid_boundary(u,v,sizeX,sizeY,b)
    else:
        return boundary_func(u,v,sizeX,sizeY,b)

@tf.function
def sampleAt(x,y, data, sizeX, sizeY, offset, d):
    '''
    Performs bilinear interpolation on point ``(x,y)`` in the grid ``data``.

    Args:
        x: A ``float`` \n
        y: A ``float`` \n
        data: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` where we want the value to be interpolated \n
        sizeX: An ``int`` representing the number of horizontal cells \n
        sizeY: An ``int`` representing the number of vertical cells \n
        offset: A ``float`` such that the value at the bottom-left corner of ``data`` is the value at point ``(offset, offset)`` \n
        d: A ``float`` representing the size of the cells of ``data``

    Returns:
        The bilinear interpolated value ``data[x,y]``
    '''
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
def advectStaggeredU(u,v,sizeX, sizeY, coords_x, coords_y, dt, offset, d):
    v_u = tf.vectorized_map(fn=lambda x:sampleAt(x[0], x[1], v, sizeX, sizeY, offset ,d), elems=(coords_x - 0.5*d, coords_y + 0.5*d))
    traced_x_u = tf.clip_by_value(coords_x - 0.5*d- dt*u, offset, offset + (sizeX-0.5)*d - 0.5*d)
    traced_y_u = tf.clip_by_value(coords_y - dt*v_u,  offset + 0.5*d, offset + (sizeY-0.5)*d)
    return tf.vectorized_map(fn=lambda x: sampleAt(x[0], x[1], u, sizeX, sizeY, offset, d), elems=(traced_x_u + 0.5*d, traced_y_u))

@tf.function
def advectStaggeredV(u, v, sizeX, sizeY, coords_x, coords_y, dt, offset, d):
    u_v = tf.vectorized_map(fn=lambda x:sampleAt(x[0], x[1], u, sizeX, sizeY, offset, d), elems=(coords_x + 0.5*d, coords_y - 0.5*d))
    traced_x_v = tf.clip_by_value(coords_x - dt*u_v, offset + 0.5*d, offset + (sizeX-0.5)*d)
    traced_y_v = tf.clip_by_value(coords_y - 0.5*d - dt*v, offset, offset + (sizeY-0.5)*d - 0.5*d)
    return tf.vectorized_map(fn=lambda x: sampleAt(x[0], x[1], v, sizeX, sizeY, offset, d), elems=(traced_x_v, traced_y_v + 0.5*d))

@tf.function
def advectStaggered(f, u,v, sizeX, sizeY, coords_x, coords_y, dt, offset, d):
    u_f = tf.vectorized_map(fn=lambda x:sampleAt(x[0], x[1], u, sizeX, sizeY, offset, d), elems=(coords_x + 0.5*d, coords_y))
    v_f = tf.vectorized_map(fn=lambda x:sampleAt(x[0], x[1], v, sizeX, sizeY, offset, d), elems=(coords_x, coords_y + 0.5*d))
    traced_x = tf.clip_by_value(coords_x - dt*u_f, offset + 0.5*d, offset + (sizeX-0.5)*d)
    traced_y = tf.clip_by_value(coords_y - dt*v_f, offset + 0.5*d, offset + (sizeY-0.5)*d)
    return tf.vectorized_map(fn=lambda x: sampleAt(x[0], x[1], f, sizeX, sizeY, offset, d), elems=(traced_x, traced_y))

@tf.function
def velocityCentered(u,v,sizeX, sizeY, coords_x, coords_y, offset, d):
    vel_x = tf.vectorized_map(fn=lambda x: sampleAt(x[0], x[1], u, sizeX, sizeY, offset, d), elems=(coords_x + 0.5*d, coords_y))
    vel_y = tf.vectorized_map(fn=lambda x: sampleAt(x[0], x[1], v, sizeX, sizeY, offset, d), elems=(coords_x, coords_y + 0.5*d))
    return vel_x, vel_y

def build_laplacian_matrix(sizeX,sizeY,a,b):
    '''
    Build a Laplacian Matrix where the diagonal is full of ``b``, and adjacent cells are equals to ``a``. Can take a bit long to execute if the grid is large.
    
    Args:
        sizeX: An ``int`` representing the number of horizontal cells
        sizeY: An ``int`` representing the number of vertical cells
        a: A ``float`` for the value of the adjacent cells
        b: A ``float`` for the value of the diagonal

    Returns:
        TO DO
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
    return tf.linalg.lu(mat)

def diffuse(f, lu, p):
    '''
    Diffuses the scalar field ``f`` using a matrix ``mat``.

    Args:
        f: A TensorFlow tensor of shape (N*N,)
        mat: A TensorFlow tensor of shape (N*N, N*N)
    Returns:
        A TensorFlow ``tensor`` of shape (N*N, 1) representing the diffused f
    '''
    if tf.rank(f).numpy() < 2:
        f = tf.expand_dims(f, 1)
    return tf.linalg.lu_solve(lu, p, f)

def solvePressure(u,v, sizeX, sizeY, h, lu, p):
    dx = tf.roll(u, shift=-1, axis=0) - u
    dy = tf.roll(v, shift=-sizeX, axis=0) - v
    div = (dx+dy)/h
    if tf.rank(div) < 2:
        div = tf.expand_dims(div, 1)
    return tf.linalg.lu_solve(lu, p, div)

def project(u,v,sizeX,sizeY, lu, q, h, boundary_func):
    p = solvePressure(u, v, sizeX, sizeY, h, lu, q)[...,0]

    gradP_u = (p - tf.roll(p, shift=1, axis=0))/h
    gradP_v = (p - tf.roll(p, shift=sizeY, axis=0))/h
    gradP_u, gradP_v = set_boundary(gradP_u, gradP_v, sizeX, sizeY, boundary_func)
    
    new_u = u - gradP_u
    new_v = v - gradP_v
    return new_u, new_v

def dissipate(s,a,dt):
    '''
    Dissipates the scalar field ``s``

    Args:
        s: A TensorFlow ``tensor`` 
        a: A ``float`` representing the dissipation rate
        dt: A ``float`` representing the time step
    '''
    return s/(1+dt*a)

def update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _lu, _p, _alpha, _vDiff_lu, _vDiff_p, _visc, _sDiff_mat_lu, _sDiff_mat_p, _kDiff, boundary_func=None, source=None, t=np.inf):
    '''
    Performs one update of the fluid simulation of the velocity field (_u,_v) and the density field _s, using Centered Grid

    Args:
        _u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _s: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _sizeX: An ``int`` representing the number of horizontal cells
        _sizeY: An ``int`` representing the number of vertical cells
        _coords_x: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the x-coordinates of the fluid's grid
        _coords_y: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the y-coordinates of the fluid's grid  
        _dt: A ``float`` representing the timestep of the simulation
        _offset: A ``float`` such that the coordinate at the bottom-left corner of the grid is ``(offset, offset)`` \n
        _h: A ``float`` representing the size of the cells of the grid
        _mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to find the irrotational vector field of the Helmholtz decomposition
        _alpha: A ``float`` representing the dissipation rate
        _vDiff_mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to diffuse velocity
        _visc: A ``float`` representing the fluid's viscosity
        _sDiff_mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to diffuse density
        _visc: A ``float`` representing the diffusion rate of the density
        boundary_func: A function of signature (``tensor``, ``tensor``, ``int``, ``int``, ``float``) -> (``tensor``, ``tensor``) where the tensors must have the same shape. Default is set to ``None``, in that case the reflexive boundaries are applied.
        source: A ``dict`` representing a density source. Default is set to ``None``, meaning there is no source
                ``source["time"]`` is an ``int`` indicating the last frame where the source exists \n
                ``source["value"]`` is a ``float`` representing its value \n
                ``source["indices"]`` is a TensorFlow ``tensor`` of shape (N,1,1) that indicates all the indices in the density grid ``_s`` where the source should be
        t: An ``int`` representing the current frame. Default is set to ``np.inf``, meaning we don't need to know the current frame number
    '''
    ## Vstep
    # advection step
    new_u = advectStaggeredU(_u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    new_v = advectStaggeredV(_u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    _u = new_u
    _v = new_v

    # diffusion step
    if _visc > 0:
        _u = diffuse(_u, _vDiff_lu, _vDiff_p)[..., 0]
        _v = diffuse(_v, _vDiff_lu, _vDiff_p)[..., 0]
        _u, _v = set_boundary(_u, _v, _sizeX, _sizeY, boundary_func)

    # projection step
    _u, _v = project(_u, _v, _sizeX, _sizeY, _lu, _p, _h, boundary_func)

    ## Sstep
    if (source is not None) and (t < source["time"]) :
        _s = addSource(_s, source["value"], source["indices"])
    # advection step
    _s = advectStaggered(_s, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)

    # diffusion step
    if _kDiff > 0:
        _s = diffuse(_s, _sDiff_mat_lu, _sDiff_mat_p)

    # dissipation step
    if _alpha > 0:
        _s = dissipate(_s, _alpha, _dt)
    
    if (source is not None) and (t < source["time"]) :
        _s = addSource(_s, source["value"], source["indices"])
    return _u, _v, _s

def simulate(n_iter, _u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, boundary_func=None, source=None, leave=True):
    '''
    Performs a fluid simulation of the velocity field ``(_u,_v)`` and the density field ``_s`` over ``n_iter`` frames

    Args:
        n_iter: An ``int`` representing the number of frames of the simulation
        _u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _s: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _sizeX: An ``int`` representing the number of horizontal cells
        _sizeY: An ``int`` representing the number of vertical cells
        _coords_x: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the x-coordinates of the fluid's grid
        _coords_y: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the y-coordinates of the fluid's grid  
        _dt: A ``float`` representing the timestep of the simulation
        _offset: A ``float`` such that the coordinate at the bottom-left corner of the grid is ``(offset, offset)`` \n
        _h: A ``float`` representing the size of the cells of the grid
        _mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to find the irrotational vector field of the Helmholtz decomposition
        _alpha: A ``float`` representing the dissipation rate
        _vDiff_mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to diffuse velocity
        _visc: A ``float`` representing the fluid's viscosity
        _sDiff_mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to diffuse density
        _kDiff: A ``float`` representing the diffusion rate of the density
        boundary_func: A function of signature (``tensor``, ``tensor``, ``int``, ``int``, ``float``) -> (``tensor``, ``tensor``) where the tensors must have the same shape. Default is set to ``None``, in that case the reflexive boundaries are applied.
        source: A ``dict`` representing a density source. Default is set to ``None``, meaning there is no source
                ``source["time"]`` is an ``int`` indicating the last frame where the source exists \n
                ``source["value"]`` is a ``float`` representing its value \n
                ``source["indices"]`` is a TensorFlow ``tensor`` of shape (N,1,1) that indicates all the indices in the density grid ``_s`` where the source should be
        leave: A bool indicating if we want or not to clear the tqdm bar at each iteration
        
    Returns:
        new_u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting the x-component of the velocity field at the end of the simulation
        new_v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting the y-component of the velocity field at the end of the simulation
        new_s: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting the density field at the end of the simulation
    '''
    for t in tqdm(range(1, n_iter+1), desc = "Simulating....", leave=leave):
        new_u, new_v, new_s = update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff,boundary_func, source, t)
        _u = new_u
        _v = new_v
        _s = new_s
    return new_u, new_v, new_s

def simulateConstrained(n_iter, _u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, keyframes=[], keyidx=[], boundary_func=None, source=None, leave=True):
    '''
    Performs a fluid simulation of the velocity field ``(_u,_v)`` and the density field ``_s`` over ``n_iter`` frames.
    This version stores intermediates states of some cells ``keyidx`` of the grid at ``keyframes``

    Args:
        n_iter: An ``int`` representing the number of frames of the simulation
        _u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _s: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting a grid of size ``(sizeX, sizeY)``
        _sizeX: An ``int`` representing the number of horizontal cells
        _sizeY: An ``int`` representing the number of vertical cells
        _coords_x: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the x-coordinates of the fluid's grid
        _coords_y: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` representing the y-coordinates of the fluid's grid  
        _dt: A ``float`` representing the timestep of the simulation
        _offset: A ``float`` such that the coordinate at the bottom-left corner of the grid is ``(offset, offset)`` \n
        _h: A ``float`` representing the size of the cells of the grid
        _mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to find the irrotational vector field of the Helmholtz decomposition
        _alpha: A ``float`` representing the dissipation rate
        _vDiff_mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to diffuse velocity
        _visc: A ``float`` representing the fluid's viscosity
        _sDiff_mat: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,sizeX*sizeY)`` representing the Laplacian Matrix used to diffuse density
        _kDiff: A ``float`` representing the diffusion rate of the density
        keyframes: A ``list`` or Numpy ``array`` of dimension 1 of ``int`` representing the frames when we want to keep track of the fluid's state
        keyidx: A ``list`` or Numpy ``array`` of dimension 1 of ``int`` representing the indices of the cells when we want to keep track of
        boundary_func: A function of signature (``tensor``, ``tensor``, ``int``, ``int``, ``float``) -> (``tensor``, ``tensor``) where the tensors must have the same shape. Default is set to ``None``, in that case the reflexive boundaries are applied.
        source: A ``dict`` representing a density source. Default is set to ``None``, meaning there is no source
                ``source["time"]`` is an ``int`` indicating the last frame where the source exists \n
                ``source["value"]`` is a ``float`` representing its value \n
                ``source["indices"]`` is a TensorFlow ``tensor`` of shape (N,1,1) that indicates all the indices in the density grid ``_s`` where the source should be
        leave: A bool indicating if we want or not to clear the tqdm bar at each iteration

    Returns:
        new_u: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting the x-component of the velocity field at the end of the simulation
        new_v: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting the y-component of the velocity field at the end of the simulation
        new_s: A TensorFlow ``tensor`` of shape ``(sizeX*sizeY,)`` reprensenting the density field at the end of the simulation
        midVel: A list of [``tensor``, ``tensor``] representing the intermediates states of some cells 
    '''

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
            _dt = tf.Variable(tf.random.uniform([1])) 

            # Gradients
            with tf.GradientTape() as tape:
                new_u = advectStaggeredU(_u, _v, _sizeX, _sizeY, _coordsX, _coordsY, _dt, _grid_min, _d)
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
            _visc = tf.Variable(tf.random.uniform([1]))

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
            _kDiff = tf.Variable(tf.random.uniform([1]))
            _dt = tf.Variable(tf.random.uniform([1]))

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
            _s = tf.Variable(tf.random.normal([_sizeX*_sizeY]), dtype=tf.float32)
            _dt = tf.Variable(tf.constant(0.025))
            _visc = tf.random.uniform([1])
            _kDiff = tf.random.uniform([1])
            _alpha = tf.random.uniform([1])
            
            # Gradients
            with tf.GradientTape() as tape:
                _sDiff_lu, _sDiff_p = build_laplacian_matrix(_sizeX, _sizeY, -_kDiff/(_d*_d), 1+4*_kDiff/(_d*_d) )
                _vDiff_lu, _vDiff_p = build_laplacian_matrix(_sizeX, _sizeY, -_visc/(_d*_d), 1+4*_visc/(_d*_d) )
                _lu, _p = build_laplacian_matrix(_sizeX, _sizeY,1/(_d*_d),-4/(_d*_d))
                new_u, new_v, new_s = update(_u, _v, _s, _sizeX, _sizeY, _coordsX, _coordsY, _dt, _grid_min, _d, _lu, _p, _alpha,  _vDiff_lu, _vDiff_p, _visc, _sDiff_lu, _sDiff_p, _kDiff)
            print("Executing OK")
            grad_solver = tape.gradient([new_s, new_u, new_v], [_u, _v, _s, _dt])

            # print(grad_solver)

            # Differentiability test
            if all([gradient is not None for gradient in grad_solver]):
                print(colored("The solver is differentiable.", 'green'))
            else:
                print(colored("The solver is not differentiable.", 'red'))











