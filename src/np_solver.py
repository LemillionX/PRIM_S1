import numpy as np
from tqdm import tqdm

def indexTo1D(i,j,sizeX):
    return j*sizeX+i

def set_solid_boundary(u,v, sizeX, sizeY, b=0):
    new_u = np.copy(u)
    new_v = np.copy(v)

    mask = np.logical_or(np.logical_or(np.equal(np.arange(sizeX*sizeY) % sizeX, 0), np.equal(np.arange(sizeX*sizeY) % sizeX, sizeX-1)),
                        np.logical_or(np.equal(np.arange(sizeX*sizeY) // sizeX, 0), np.equal(np.arange(sizeX*sizeY) // sizeX, sizeY-1)))
    indices = np.where(mask)
    new_u[indices] = b
    new_v[indices] = b
    return new_u, new_v

def set_boundary(u,v, sizeX,sizeY,boundary_func=None,b=0):
    if boundary_func is None:
        return set_solid_boundary(u,v,sizeX,sizeY,b)
    else:
        return boundary_func(u,v,sizeX,sizeY,b)
   
def sampleAt(x,y,data,sizeX, sizeY, offset,d):
    _x = (x-offset)/d - 0.5
    _y = (y-offset)/d - 0.5
    i0 = np.clip(np.floor(_x),0,sizeX-1)
    j0 = np.clip(np.floor(_y),0,sizeY-1)
    i1 = np.clip(i0+1,0,sizeY-1)
    j1 = np.clip(j0+1,0,sizeY-1)

    p00 = data[ int(indexTo1D(i0,j0,sizeX))]
    p01 = data[ int(indexTo1D(i0,j1,sizeX))]
    p10 = data[ int(indexTo1D(i1,j0,sizeX))]
    p11 = data[ int(indexTo1D(i1,j1,sizeX))]
    
    t_i0 = (offset + (i1+0.5)*d -x)/d
    t_j0 = (offset + (j1+0.5)*d -y)/d
    t_i1 = (x - (offset + (i0+0.5)*d))/d
    t_j1 = (y - (offset + (j0+0.5)*d))/d

    return t_i0*t_j0*p00 + t_i0*t_j1*p01 + t_i1*t_j0*p10 + t_i1*t_j1*p11

def advectCentered(f,u,v,sizeX,sizeY,coords_x,coords_y,dt,offset,d):
    traced_x = np.clip(coords_x - dt*u, offset+0.5*d, offset+(sizeX-0.5)*d)
    traced_y = np.clip(coords_y - dt*v, offset+0.5*d, offset+(sizeY-0.5)*d)
    new_grid = []
    for it in range(len(traced_x)):
        new_grid.append(sampleAt(traced_x[it],traced_y[it],f,sizeX,sizeY,offset,d))
    return np.array(new_grid)

def build_laplacian_matrix(sizeX,sizeY,a,b):
    mat = np.zeros((sizeX*sizeY, sizeX*sizeY))
    for it in range(sizeX*sizeY):
        i = it%sizeX
        j = it//sizeX
        mat[it,it]=b
        if (i>0):
            mat[it,it-1]=a
        if (i<sizeX-1):
            mat[it,it+1]=a
        if (j>0):
            mat[it,it-sizeX]=a
        if (j<sizeY-1):
            mat[it,it+sizeX]=a
    return mat

def diffuse(f,mat):
    return np.linalg.solve(mat,f)

def solvePressure(u,v,sizeX,sizeY,h,mat):
    dx = np.roll(u, shift=-1, axis=0) - np.roll(u, shift=1, axis=0)
    dy = np.roll(v, shift=-sizeX, axis=0) - np.roll(v, shift=sizeX, axis=0)
    div = (dx+dy)*0.5/h
    return np.linalg.solve(mat,div)

def project(u,v,sizeX,sizeY,mat,h,boundary_func):
    _u,_v = set_boundary(u,v, sizeX, sizeY,boundary_func)
    p = solvePressure(_u,_v,sizeX,sizeY,h, mat)
    gradP_u = (0.5/h)*(np.roll(p, shift=-1, axis=0) - np.roll(p, shift=1, axis=0))
    gradP_v = (0.5/h)*(np.roll(p, shift=-sizeX, axis=0) - np.roll(p, shift=sizeX, axis=0))
    new_u = _u - gradP_u
    new_v = _v - gradP_v
    return set_boundary(new_u, new_v, sizeX, sizeY, boundary_func)

def dissipate(s,a,dt):
    return s/(1+dt*a)

def update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, boundary_func=None):
    ## Vstep
    # advection step
    new_u = advectCentered(_u, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    new_v = advectCentered(_v, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    _u = new_u
    _v = new_v

    # diffusion step
    if _visc > 0:
        _u = diffuse(_u, _vDiff_mat)
        _v = diffuse(_v, _vDiff_mat)

    # projection step
    _u, _v = project(_u, _v, _sizeX, _sizeY, _mat, _h, boundary_func)


    ## Sstep
    # advection step
    _s = advectCentered(_s, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    
    # diffusion step
    if _kDiff > 0:
        _s = diffuse(_s, _sDiff_mat)

    # dissipation step
    _s = dissipate(_s, _alpha, _dt)

    return _u, _v, _s

def simulate(n_iter, _u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, boundary_func=None):
    for _ in tqdm(range(1, n_iter+1), desc = "Simulating...."):
        new_u, new_v, new_s = update(_u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff, boundary_func)
        _u = new_u
        _v = new_v
        _s = new_s
    return new_u, new_v, new_s

