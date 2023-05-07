import tensorflow as tf
import tf_solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

@tf.function
def loss_quadratic(current, target, currentMidVel=[], midVel=[], weights=[]):
    density_loss = 0.5*(tf.norm(current - target)**2)
    velocity_loss = tf.constant(0, dtype=tf.float32) 
    for t in range(len(midVel)): # iterate over keyframes
        velocity_loss += weights[t]*(1.0+tf.keras.losses.cosine_similarity(tf.reshape(tf.convert_to_tensor(midVel[t]), [-1]),  tf.convert_to_tensor(currentMidVel[t])))
        # for i in range(len(midVel[t])): # iterate over dimension
            # velocity_loss += 0.5*weights[t]*(tf.norm(currentMidVel[t][i] - midVel[t][i]))**2
    return density_loss + velocity_loss, density_loss, velocity_loss

def train(_max_iter, _d_init, _target, _nFrames, _u_init, _v_init, _fluidSettings, _coordsX, _coordsY, _boundary, filename, constraint=None, learning_rate=1.1, debug=False):
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
    velocity_field_x, velocity_field_y = slv.set_boundary(tf.convert_to_tensor(_u_init, dtype=tf.float32),tf.convert_to_tensor(_v_init, dtype=tf.float32), sizeX, sizeY, _boundary)
    density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
    trained_vel_x = tf.identity(velocity_field_x)
    trained_vel_y = tf.identity(velocity_field_y)
    dt = tf.convert_to_tensor(timestep, dtype=tf.float32)
    coords_X = tf.convert_to_tensor(_coordsX, dtype=tf.float32)
    coords_Y = tf.convert_to_tensor(_coordsY, dtype=tf.float32)

    ## Initial guess
    loss, d_loss, v_loss = loss_quadratic(density_field, target_density)
    with tf.GradientTape() as tape:
        velocity_field_x = tf.Variable(velocity_field_x)
        velocity_field_y = tf.Variable(velocity_field_y)
        _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, keyframes, keyidx, _boundary, source, leave=False)
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
            _,_, density_field, midVel = slv.simulateConstrained(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, keyframes, keyidx, _boundary, source, leave=False)
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
    trained_a = tf.identity(a_init)
    dt = tf.convert_to_tensor(timestep, dtype=tf.float32)
    coords_X = tf.convert_to_tensor(_coordsX, dtype=tf.float32)
    coords_Y = tf.convert_to_tensor(_coordsY, dtype=tf.float32)

   ## Initial guess
    loss, d_loss, v_loss = loss_quadratic(density_field, target_density)
    with tf.GradientTape() as tape:
        trained_a = tf.Variable(trained_a)
        _u_init, _v_init = slv.curl2Dvector(trained_a, sizeX)
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
            _u_init, _v_init = slv.curl2Dvector(trained_a, sizeX)
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
    _u_init, _v_init = slv.curl2Dvector(trained_a, sizeX)
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