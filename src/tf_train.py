import tensorflow as tf
import tf_solver_staggered as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

@tf.function
def loss_quadratic(current, target, currentMidVel=[], midVel=[], weights=[]):
    density_loss = 0.5*(tf.norm(current - target)**2)
    velocity_loss = tf.reduce_sum(tf.multiply(weights, 1.0 + tf.keras.losses.cosine_similarity(midVel, currentMidVel)))
    return density_loss + velocity_loss, density_loss, velocity_loss


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

def train(_max_iter, _d_init, _target, _nFrames, _u_init, _v_init, _fluidSettings, _coordsX, _coordsY, filename, constraint=None, learning_rate=1.1, debug=False):
    sizeX = int(np.sqrt(len(_target)))          # number of elements in the x-axis
    sizeY = int(np.sqrt(len(_target)))           # number of elements in the y-axis
    assert (sizeX == sizeY), "Dimensions on axis are different !"

    timestep = _fluidSettings["timestep"]
    grid_min = _fluidSettings["grid_min"]
    grid_max = _fluidSettings["grid_max"]
    h = (grid_max-grid_min)/sizeX
    k_diff = _fluidSettings["diffusion_coeff"]
    alpha = _fluidSettings["dissipation_rate"]
    visc = _fluidSettings["viscosity"]
    _boundary = _fluidSettings["boundary"]
    source = _fluidSettings["source"]
    keyframes = []
    keyidx = []
    keyvalues = []
    key_weights = []
    if constraint is not None:
        keyframes = constraint["keyframes"]
        keyidx = constraint["indices"]
        keyvalues = tf.reshape(tf.convert_to_tensor(constraint["values"], dtype=tf.float32), (-1, 2*len(constraint["values"][0][0])))
        key_weights = constraint["weights"]

    ## Pre-build matrices
    laplace_mat_LU, laplace_mat_P = slv.build_laplacian_matrix(sizeX, sizeY, 1/( h*h), -4/( h*h), _boundary)
    velocity_diff_LU, velocity_diff_P = slv.build_laplacian_matrix(sizeX, sizeY, -visc*timestep/( h*h), 1+4*visc*timestep/( h*h))
    scalar_diffuse_LU, scalar_diffuse_P = slv.build_laplacian_matrix(sizeX, sizeY, -k_diff*timestep/( h*h), 1+4*k_diff*timestep/( h*h) )

    ## Converting variables to tensors
    target_density = tf.convert_to_tensor(_target, dtype=tf.float32)
    velocity_field_x, velocity_field_y = tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32)
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    trained_vel_x, trained_vel_y =  slv.project(velocity_field_x, velocity_field_y, sizeX, sizeY, laplace_mat_LU, laplace_mat_P, h, _boundary) 
    dt = tf.convert_to_tensor(timestep, dtype=tf.float32)
    coords_X = tf.convert_to_tensor(_coordsX, dtype=tf.float32)
    coords_Y = tf.convert_to_tensor(_coordsY, dtype=tf.float32)

    ## Initial guess
    loss, d_loss, v_loss = loss_quadratic(density_field, target_density)
    with tf.GradientTape() as tape:
        velocity_field_x = tf.Variable(trained_vel_x)
        velocity_field_y = tf.Variable(trained_vel_y)
        _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat_LU, laplace_mat_P, alpha, velocity_diff_LU, velocity_diff_P, visc, scalar_diffuse_LU, scalar_diffuse_P, k_diff, keyframes, keyidx, _boundary, source, leave=False)
        loss,  d_loss, v_loss = loss_quadratic(density_field, target_density, midVel, keyvalues, key_weights)
    grad = tape.gradient([loss], [velocity_field_x, velocity_field_y])
    print("[step 0] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(l_rate=0, loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=tf.norm(grad[:2]).numpy()))

    ## Optimisation
    count = 0
    while (count < _max_iter and loss > 0.1 and tf.norm(grad[:2]).numpy() > 1e-04):
        # l_rate = learning_rate*abs(tf.random.normal([1]))
        # l_rate = tf.constant(learning_rate/np.sqrt(count+1,),dtype = tf.float32)
        l_rate = tf.constant(learning_rate, dtype = tf.float32)
        density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
        trained_vel_x = trained_vel_x - l_rate*grad[0]
        trained_vel_y = trained_vel_y - l_rate*grad[1]
        with tf.GradientTape() as tape:
            velocity_field_x = tf.Variable(trained_vel_x)
            velocity_field_y = tf.Variable(trained_vel_y)
            velocity_field_x, velocity_field_y = slv.project(velocity_field_x, velocity_field_y, sizeX, sizeY, laplace_mat_LU, laplace_mat_P, h, _boundary) 
            _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat_LU, laplace_mat_P, alpha, velocity_diff_LU, velocity_diff_P, visc, scalar_diffuse_LU, scalar_diffuse_P, k_diff, keyframes, keyidx, _boundary, source, leave=False)
            loss, d_loss, v_loss = loss_quadratic(density_field, target_density, midVel, keyvalues, key_weights)
        count += 1
        grad = tape.gradient([loss], [velocity_field_x, velocity_field_y])
        # print(grad)
        if (count < 3) or (count%10 == 0):
            if debug:
                print(midVel)
            print("[step {count}] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(count=count, l_rate=l_rate.numpy(),loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=tf.norm(grad[:2]).numpy()))

    if (count < _max_iter and count > 0):
        print("[step {count}] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(count=count, l_rate=l_rate.numpy(),loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=tf.norm(grad[:2]).numpy()))


    if debug:
        print("After {count} iterations, the velocity field is ".format(count=count))
        print("x component = ")
        print(trained_vel_x)
        print("y component = ")
        print(trained_vel_y)
        print(" and gradient norm = {norm}".format(norm = tf.norm(grad[:2]).numpy()))

    ## Testing
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    velocity_field_x = trained_vel_x
    velocity_field_y = trained_vel_y

    ## Plot Animation
    output_dir = "output"
    v_path, d_path, resolution_limit = viz.init_dir(output_dir, filename, sizeX, sizeX+1) 
    fps = 20
    fig, ax, Q = viz.init_viz(velocity_field_x,velocity_field_y,density_field, coords_X, coords_Y, sizeX, sizeY, grid_min, h, v_path, d_path, resolution_limit)

    pbar = tqdm(range(1, 2*_nFrames+1), desc = "Simulating....")

    for t in pbar:
        velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h,  laplace_mat_LU, laplace_mat_P, alpha, velocity_diff_LU, velocity_diff_P, visc, scalar_diffuse_LU, scalar_diffuse_P, k_diff, _boundary, source, t)
        # Viz update
        viz.draw_density(np.flipud(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy()), os.path.join(d_path, '{:04d}.png'.format(t)))
        u_viz, v_viz = viz.draw_velocity(velocity_field_x,velocity_field_y, sizeX, sizeY, coords_X, coords_Y, grid_min, h)
        Q.set_UVC(u_viz,v_viz)
        plt.savefig(os.path.join(v_path, '{:04d}'.format(t)))

    viz.frames2gif(v_path, v_path+".gif", fps)
    viz.frames2gif(d_path, d_path+".gif", fps)
    return trained_vel_x, trained_vel_y


def train_scalar_field(_max_iter, _d_init, _target, _nFrames, a_init, _fluidSettings, _coordsX, _coordsY, _boundary, filename, constraint=None, learning_rate=1.1, debug=False):
    sizeX = int(np.sqrt(len(_target)))          # number of elements in the x-axis
    sizeY = int(np.sqrt(len(_target)))           # number of elements in the y-axis
    assert (sizeX == sizeY), "Dimensions on axis are different !"


    timestep = _fluidSettings["timestep"]
    grid_min = _fluidSettings["grid_min"]
    grid_max = _fluidSettings["grid_max"]
    h = (grid_max-grid_min)/sizeX
    k_diff = _fluidSettings["diffusion_coeff"]
    alpha = _fluidSettings["dissipation_rate"]
    visc = _fluidSettings["viscosity"]
    source = _fluidSettings["source"]
    keyframes = []
    keyidx = []
    keyvalues = []
    key_weights = []
    if constraint is not None:
        keyframes = constraint["keyframes"]
        keyidx = constraint["indices"]
        keyvalues = [ [tf.convert_to_tensor(u[0], dtype=tf.float32), tf.convert_to_tensor(u[1], dtype=tf.float32)] for u in constraint["values"]]
        key_weights = constraint["weights"]

     ## Pre-build matrices
    laplace_mat =  tf.convert_to_tensor(slv.build_laplacian_matrix(sizeX, sizeY, 1/( h*h), -4/( h*h)), dtype=tf.float32)
    velocity_diff_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(sizeX, sizeY, -visc*timestep/( h*h), 1+4*visc*timestep/( h*h) ), dtype=tf.float32)
    scalar_diffuse_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(sizeX, sizeY, -k_diff*timestep/( h*h), 1+4*k_diff*timestep/( h*h) ), dtype=tf.float32)

    ## Converting variables to tensors
    target_density = tf.convert_to_tensor(_target, dtype=tf.float32)
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    trained_a = tf.convert_to_tensor(a_init, dtype=tf.float32)
    dt = tf.convert_to_tensor(timestep, dtype=tf.float32)
    coords_X = tf.convert_to_tensor(_coordsX, dtype=tf.float32)
    coords_Y = tf.convert_to_tensor(_coordsY, dtype=tf.float32)

   ## Initial guess
    loss, d_loss, v_loss = loss_quadratic(density_field, target_density)
    with tf.GradientTape() as tape:
        trained_a = tf.Variable(trained_a)
        _u_init, _v_init = slv.curl2Dvector(trained_a, sizeX, h)
        velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)
        _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, keyframes, keyidx, _boundary, source, leave=False)
        loss,  d_loss, v_loss = loss_quadratic(density_field, target_density, midVel, keyvalues, key_weights)
    grad = tape.gradient([loss], [trained_a])
    print("[step 0] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(l_rate=0, loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=tf.norm(grad[:2]).numpy()))

    ## Optimisation
    count = 0
    while (count < _max_iter and loss > 0.1 and tf.norm(grad[:2]).numpy() > 1e-04):
        # l_rate = learning_rate*abs(tf.random.normal([1]))
        # l_rate = tf.constant(learning_rate/np.sqrt(count+1,),dtype = tf.float32)
        l_rate = tf.constant(learning_rate, dtype = tf.float32)
        density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
        trained_a = trained_a - l_rate*grad[0]
        with tf.GradientTape() as tape:
            trained_a = tf.Variable(trained_a)
            _u_init, _v_init = slv.curl2Dvector(trained_a, sizeX, h)
            velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)
            _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, keyframes, keyidx, _boundary, source, leave=False)
            loss, d_loss, v_loss = loss_quadratic(density_field, target_density, midVel, keyvalues, key_weights)
        count += 1
        grad = tape.gradient([loss], [trained_a])
        # print(grad)
        if (count < 3) or (count%10 == 0):
            if debug:
                print(midVel)
            print("[step {count}] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(count=count, l_rate=l_rate.numpy(),loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=tf.norm(grad[:2]).numpy()))

    if (count < _max_iter and count > 0):
        print("[step {count}] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(count=count, l_rate=l_rate.numpy(),loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=tf.norm(grad[:2]).numpy()))

    ## Testing
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    _u_init, _v_init = slv.curl2Dvector(trained_a, sizeX, h)
    velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)

    ## Plot initialisation 
    x,y = np.meshgrid(coords_X[:sizeX], coords_Y[::sizeX])
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    Q = ax.quiver(x, y, tf.reshape(velocity_field_x, shape=(sizeX, sizeY)).numpy(), tf.reshape(velocity_field_y, shape=(sizeX, sizeY)).numpy(), color='red', scale_units='width')

    ## Plot Animation
    output_dir = "output"
    velocity_name = filename["velocity"]
    density_name = filename["density"]
    dir_path = os.path.join(os.getcwd().rsplit("\\",1)[0], output_dir)
    save_path =  os.path.join(dir_path, velocity_name)
    fps = 20
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    else:
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Error deleting file: {file_path} - {e}')
    if not os.path.isdir(os.path.join(dir_path, density_name)):
        os.mkdir(os.path.join(dir_path, density_name))
    else:
        for file in os.listdir(os.path.join(dir_path, density_name)):
            file_path = os.path.join(os.path.join(dir_path, density_name), file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Error deleting file: {file_path} - {e}')

    print("Images will be saved here:", save_path)
    pbar = tqdm(range(1, _nFrames*2+1), desc = "Simulating....")
    plt.savefig(os.path.join(save_path, '{:04d}'.format(0)))
    viz.draw_density(np.flipud(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy()), os.path.join(dir_path, density_name, '{:04d}.png'.format(0)))

    for t in pbar:
        velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, _boundary, source, t)
        # Viz update
        viz.draw_density(np.flipud(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy()), os.path.join(dir_path, density_name, '{:04d}.png'.format(t)))
        u_viz = tf.reshape(velocity_field_x, shape=(sizeX, sizeY)).numpy()
        v_viz = tf.reshape(velocity_field_y, shape=(sizeX, sizeY)).numpy()
        Q.set_UVC(u_viz,v_viz)
        plt.savefig(os.path.join(save_path, '{:04d}'.format(t)))

    viz.frames2gif(os.path.join(dir_path, velocity_name), os.path.join(dir_path, velocity_name+".gif"), fps)
    viz.frames2gif(os.path.join(dir_path, density_name), os.path.join(dir_path, density_name+".gif"), fps)
    return trained_a

def train_vortices(_max_iter, _d_init, _target, _nFrames, n_vortices, centers_init, radius_init, w_init, _fluidSettings, _coordsX, _coordsY, _boundary, filename, constraint=None, learning_rate=1.1, debug=False):
    sizeX = int(np.sqrt(len(_target)))          # number of elements in the x-axis
    sizeY = int(np.sqrt(len(_target)))           # number of elements in the y-axis
    assert (sizeX == sizeY), "Dimensions on axis are different !"


    timestep = _fluidSettings["timestep"]
    grid_min = _fluidSettings["grid_min"]
    grid_max = _fluidSettings["grid_max"]
    h = (grid_max-grid_min)/sizeX
    k_diff = _fluidSettings["diffusion_coeff"]
    alpha = _fluidSettings["dissipation_rate"]
    visc = _fluidSettings["viscosity"]
    source = _fluidSettings["source"]
    keyframes = []
    keyidx = []
    keyvalues = []
    key_weights = []
    if constraint is not None:
        keyframes = constraint["keyframes"]
        keyidx = constraint["indices"]
        keyvalues = [ [tf.convert_to_tensor(u[0], dtype=tf.float32), tf.convert_to_tensor(u[1], dtype=tf.float32)] for u in constraint["values"]]
        key_weights = constraint["weights"]

     ## Pre-build matrices
    laplace_mat =  tf.convert_to_tensor(slv.build_laplacian_matrix(sizeX, sizeY, 1/( h*h), -4/( h*h)), dtype=tf.float32)
    velocity_diff_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(sizeX, sizeY, -visc*timestep/( h*h), 1+4*visc*timestep/( h*h) ), dtype=tf.float32)
    scalar_diffuse_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(sizeX, sizeY, -k_diff*timestep/( h*h), 1+4*k_diff*timestep/( h*h) ), dtype=tf.float32)

    ## Converting variables to tensors
    target_density = tf.convert_to_tensor(_target, dtype=tf.float32)
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    dt = tf.convert_to_tensor(timestep, dtype=tf.float32)
    coords_X = tf.convert_to_tensor(_coordsX, dtype=tf.float32)
    coords_Y = tf.convert_to_tensor(_coordsY, dtype=tf.float32)
    coords = tf.stack((coords_X, coords_Y), axis=1)
    trained_centers = tf.identity(centers_init)
    trained_radius = tf.identity(radius_init)
    trained_w = tf.identity(w_init)

   ## Initial guess
    loss, d_loss, v_loss = loss_quadratic(density_field, target_density)
    with tf.GradientTape() as tape:
        trained_centers = tf.Variable(trained_centers)
        trained_radius = tf.Variable(trained_radius)
        trained_w = tf.Variable(trained_w)
        _u_init, _v_init = init_vortices(n_vortices, trained_centers, trained_radius, trained_w, coords, sizeX)
        velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)
        _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, keyframes, keyidx, _boundary, source, leave=False)
        loss,  d_loss, v_loss = loss_quadratic(density_field, target_density, midVel, keyvalues, key_weights)
    grad = tape.gradient([loss], [trained_centers, trained_radius, trained_w])
    print("[step 0] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(l_rate=0, loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=(tf.norm(grad[0]) + tf.norm(grad[1]) + tf.norm(grad[2])).numpy()))

    ## Optimisation
    count = 0
    while (count < _max_iter and loss > 0.1 and (tf.norm(grad[0]) + tf.norm(grad[1]) + tf.norm(grad[2])).numpy() > 1e-04):
        # l_rate = learning_rate*abs(tf.random.normal([1]))
        # l_rate = tf.constant(learning_rate/np.sqrt(count+1,),dtype = tf.float32)
        l_rate = tf.constant(learning_rate, dtype = tf.float32)
        density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
        trained_centers = trained_centers - l_rate*grad[0]
        trained_radius = trained_radius - l_rate*grad[1]
        trained_w = trained_w - l_rate*grad[2]
        with tf.GradientTape() as tape:
            trained_centers = tf.Variable(trained_centers)
            trained_radius = tf.Variable(trained_radius)
            trained_w = tf.Variable(trained_w)
            _u_init, _v_init = init_vortices(n_vortices, trained_centers, trained_radius, trained_w, coords, sizeX)
            velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)
            _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, keyframes, keyidx, _boundary, source, leave=False)
            loss, d_loss, v_loss = loss_quadratic(density_field, target_density, midVel, keyvalues, key_weights)
        count += 1
        grad = tape.gradient([loss], [trained_centers, trained_radius, trained_w])
        # print(grad)
        if (count < 3) or (count%10 == 0):
            if debug:
                print(midVel)
            print("[step {count}] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(count=count, l_rate=l_rate.numpy(),loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=(tf.norm(grad[0]) + tf.norm(grad[1]) + tf.norm(grad[2])).numpy()))

    if (count < _max_iter and count > 0):
        print("[step {count}] : learning_rate = {l_rate:f}, loss = {loss:f}, density_loss = {d_loss:f}, velocity_loss = {v_loss:f}, gradient_norm = {g_norm:f}".format(count=count, l_rate=l_rate.numpy(),loss=loss.numpy(), d_loss=d_loss, v_loss=v_loss, g_norm=(tf.norm(grad[0]) + tf.norm(grad[1]) + tf.norm(grad[2])).numpy()))

    ## Testing
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    _u_init, _v_init = init_vortices(n_vortices, trained_centers, trained_radius, trained_w, coords, sizeX)
    velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)

    ## Plot initialisation 
    x,y = np.meshgrid(coords_X[:sizeX], coords_Y[::sizeX])
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    Q = ax.quiver(x, y, tf.reshape(velocity_field_x, shape=(sizeX, sizeY)).numpy(), tf.reshape(velocity_field_y, shape=(sizeX, sizeY)).numpy(), color='red', scale_units='width')

    ## Plot Animation
    output_dir = "output"
    velocity_name = filename["velocity"]
    density_name = filename["density"]
    dir_path = os.path.join(os.getcwd().rsplit("\\",1)[0], output_dir)
    save_path =  os.path.join(dir_path, velocity_name)
    fps = 20
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    else:
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Error deleting file: {file_path} - {e}')
    if not os.path.isdir(os.path.join(dir_path, density_name)):
        os.mkdir(os.path.join(dir_path, density_name))
    else:
        for file in os.listdir(os.path.join(dir_path, density_name)):
            file_path = os.path.join(os.path.join(dir_path, density_name), file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Error deleting file: {file_path} - {e}')

    print("Images will be saved here:", save_path)
    pbar = tqdm(range(1, _nFrames*2+1), desc = "Simulating....")
    plt.savefig(os.path.join(save_path, '{:04d}'.format(0)))
    viz.draw_density(np.flipud(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy()), os.path.join(dir_path, density_name, '{:04d}.png'.format(0)))

    for t in pbar:
        velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, _boundary, source, t)
        # Viz update
        viz.draw_density(np.flipud(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy()), os.path.join(dir_path, density_name, '{:04d}.png'.format(t)))
        u_viz = tf.reshape(velocity_field_x, shape=(sizeX, sizeY)).numpy()
        v_viz = tf.reshape(velocity_field_y, shape=(sizeX, sizeY)).numpy()
        Q.set_UVC(u_viz,v_viz)
        plt.savefig(os.path.join(save_path, '{:04d}'.format(t)))

    viz.frames2gif(os.path.join(dir_path, velocity_name), os.path.join(dir_path, velocity_name+".gif"), fps)
    viz.frames2gif(os.path.join(dir_path, density_name), os.path.join(dir_path, density_name+".gif"), fps)
    return trained_centers, trained_radius, trained_w
