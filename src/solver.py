import numpy as np

def addForce(u0, f, dt):
    u0 += dt*f
    return u0

def set_solid_boundary(grid, b=0):
    grid[0, :] = b
    grid[-1,:] = b
    grid[:, 0] = b
    grid[:, -1] = b
    return grid


def periodic_bound(i, dim):
    return i%dim

def fixed_bound(i, dim):
    return np.maximum(0,np.minimum(i, dim-1))

def bounds(i,dim, type = "fixed"):
    if type=="periodic":
        return periodic_bound(i, dim)

    if type =="fixed":
        return fixed_bound(i,dim)
    

def interpField(pos, field, origin, voxel_size, debug=False):
    dim_y, dim_x = field.shape

    j0 = int(np.floor((pos - origin)[0]/voxel_size - 0.5))
    i0 = int(np.floor((pos - origin)[1]/voxel_size - 0.5))
    
    tx = np.array([1 - (pos[0] -  (origin[0] + (j0+0.5)*voxel_size))/voxel_size, (pos[0] -  (origin[0] + (j0+0.5)*voxel_size))/voxel_size])
    ty = np.array([1 - (pos[1] - (origin[1] + (i0+0.5)*voxel_size))/voxel_size,  (pos[1] -  (origin[1] + (i0+0.5)*voxel_size))/voxel_size])
    value = 0
    for j in range(2):
        for i in range(2):
            value += tx[j]*ty[i]*field[bounds(i0+i, dim_y), bounds(j0+j, dim_x)]
    if debug:
        # print(tx, ty)
        # print(value)
        if np.any((tx < -0.00001)|(tx > 1.00001 )) or np.any((ty < -0.00001)|(ty > 1.00001 )):
            print("There seems to be an error...")
            print(i0, j0)
            print(pos)
            print(tx, ty)
            for j in range(2):
                for i in range(2):
                    print(field[bounds(i0+i, dim_y), bounds(j0+j, dim_x)])
            print("value = ", value)
            wait = input("Press Enter to continue.")
    return value


def advect(u0, field, dt, coords, origin, voxel_size, grid_max, grid_min, debug=False):
    dim_y, dim_x, N_DIM = coords.shape #i corresponds to y-axis, j to x_axis
    u1 = np.zeros_like(u0)
    for i in range(dim_y):
        for j in range(dim_x):
            traced_x = np.clip(coords[i,j] - dt*field[i,j], grid_min, grid_max)
            # traced_x = coords[i,j] - dt*field[i,j]
            if debug and traced_x[1] > grid_max:
                print("x = ", coords[i,j])
                print(field[i,j])
                print("traced_x = ", traced_x)
            u1[i,j] = interpField(traced_x, u0, origin, voxel_size, debug)
    u1 = set_solid_boundary(u1)
    return u1

def linear_solver(u1, u0, a, b, n_iter, bound = 0):
    dim_x, dim_y = u1.shape
    for it in range(n_iter):
        # u_temp = np.copy(u1) 
        for i in range(1,dim_x-1):
            for j in range(1,dim_y-1):
                # u1[i,j] = (u0[i,j] + a*(u_temp[i+1, j] + u_temp[i, j+1] + u_temp[i-1, j] + u_temp[i,j-1]))/b
                u1[i,j] = (u0[i,j] + a*(u1[i+1, j] + u1[i, j+1] + u1[i-1, j] + u1[i,j-1]))/b
        u1 = set_solid_boundary(u1, bound)
    return u1

def diffuse(u1, u0, visc, dt, D, n_iter, bound=0):
    a = visc*dt/(D*D)
    if a > 0.0:
        return linear_solver(u1, u0, a, 1+4*a, n_iter, bound)
    else:
        return u0

def project(u1, u0, D, n_iter):
    # compute divergence
    dim_x, dim_y, n_dim = u0.shape
    div = np.zeros((dim_x, dim_y))
    q = np.zeros((dim_x, dim_y))
    for i in range(1,dim_x-1):
        for j in range(1,dim_y-1):
            div[i,j] = 0.5*(u0[i, j+1, 0] - u0[i, j-1, 0] + u0[i+1, j, 1] - u0[i-1, j, 1])*D
    div = set_solid_boundary(div)
    q = set_solid_boundary(q)


    # solve the system
    q = linear_solver(q, div, -1, -4, n_iter)
    error = test_projection(q,u0, D)
    #print("Projection error = ", error)
    # print("\n div : \n", div)
    # print("q : \n", q)

    # update
    for i in range(1,dim_x-1):
        for j in range(1, dim_y-1):
            u1[i,j,0] = u0[i,j,0] - 0.5*(q[i,j+1] - q[i, j-1])/D
            u1[i,j,1] = u0[i,j,1] - 0.5*(q[i+1, j] - q[i-1, j] )/D
    for d in range(n_dim):
        u1[:,:,d] = set_solid_boundary(u1[:,:,d])
    return u1


def test_projection(q, u, D):
    dim_x, dim_y = q.shape
    laplace_q = np.zeros((dim_x, dim_y))
    div_u = np.zeros((dim_x, dim_y))
    for i in range(1, dim_x-1):
        for j in range(1, dim_y-1):
            laplace_q[i,j] = (q[i, j+1] + q[i, j-1] + q[i+1, j] + q[i-1, j] - 4*q[i,j])/(D**2)
            div_u[i,j] = 0.5*(u[i,j+1,0] - u[i,j-1,0] + u[i+1, j, 1] - u[i-1, j, 1])/D
    return np.linalg.norm(laplace_q - div_u)/np.linalg.norm(div_u)

def dissipate(s, a, dt):
    return s/(1+dt*a)

if __name__ in ["__main__", "__builtin__"]:
    print("Hello world")
    field = np.ones((3,3))
    field[1,1] = 0
    pos = np.array([1.75, 1.75])
    print(interpField(pos, field))