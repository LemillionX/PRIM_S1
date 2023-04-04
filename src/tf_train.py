import tensorflow as tf
import tf_solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def loss_quadratic(current, target):
    return 0.5*(tf.norm(current - target)**2)

def train(_max_iter, _d_init, _target, _nFrames, _u_init, _v_init, _fluidSettings, _coordsX, _coordsY, _boundary, filename, debug=False):
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
    loss = loss_quadratic(density_field, target_density)
    with tf.GradientTape() as tape:
        velocity_field_x = tf.Variable(velocity_field_x)
        velocity_field_y = tf.Variable(velocity_field_y)
        _, _, density_field = slv.simulate(_nFrames, velocity_field_x, velocity_field_y, density_field ,sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, _boundary, leave=False)
        loss = loss_quadratic(density_field, target_density)
    grad = tape.gradient([loss], [velocity_field_x, velocity_field_y])
    print("[step 0] : loss = ", loss.numpy(),  ", gradient norm = ",tf.norm(grad).numpy())

    ## Optimisation
    count = 0
    while (count < _max_iter and loss > 0.1 and tf.norm(grad).numpy() > 1e-04):
        # learning_rate = 10*abs(tf.random.normal([1]))
        learning_rate = tf.constant(10.1/np.sqrt(count+1,),dtype = tf.float32)
        # learning_rate = tf.constant(25,dtype = tf.float32)
        density_field = tf.convert_to_tensor(_d_init, dtype=tf.float32)
        trained_vel_x = trained_vel_x - learning_rate*grad[0]
        trained_vel_y = trained_vel_y - learning_rate*grad[1]
        with tf.GradientTape() as tape:
            velocity_field_x = tf.Variable(trained_vel_x)
            velocity_field_y = tf.Variable(trained_vel_y)
            _,_, density_field = slv.simulate(_nFrames, velocity_field_x, velocity_field_y, density_field, sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, _boundary, leave=False)
            loss = loss_quadratic(density_field, target_density)
        count += 1
        grad = tape.gradient([loss], [velocity_field_x, velocity_field_y])
        # print(grad)
        if (count < 3) or (count%10 == 0):
            print("[step", count, "] : learning_rate = ", learning_rate.numpy(), ", loss = ", loss.numpy(), ", gradient norm = ", tf.norm(grad).numpy())

    if (count < _max_iter and count > 0):
        print("[step", count, "] : learning_rate = ", learning_rate.numpy(), ", loss = ", loss.numpy(), ", gradient norm = ", tf.norm(grad).numpy())

    if debug:
        print("After ", count, " iterations, the velocity field is " )
        print("x component = ")
        print(trained_vel_x)
        print("y component = ")
        print(trained_vel_y)
        print(" and gradient norm = ",tf.norm(grad).numpy())

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
    if not os.path.isdir(os.path.join(dir_path, density_name)):
        os.mkdir(os.path.join(dir_path, density_name))

    print("Images will be saved here:", save_path)
    pbar = tqdm(range(1, _nFrames*2+1), desc = "Simulating....")
    plt.savefig(os.path.join(save_path, '{:04d}'.format(0)))
    viz.draw_density(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy(), os.path.join(dir_path, density_name, '{:04d}.png'.format(0)))

    for t in pbar:
        velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,sizeX, sizeY, coords_X, coords_Y, dt, grid_min, h, laplace_mat, alpha, velocity_diff_mat, visc, scalar_diffuse_mat, k_diff, _boundary)
        # Viz update
        viz.draw_density(tf.reshape(density_field, shape=(sizeX, sizeY)).numpy(), os.path.join(dir_path, density_name, '{:04d}.png'.format(t)))
        u_viz = tf.reshape(velocity_field_x, shape=(sizeX, sizeY)).numpy()
        v_viz = tf.reshape(velocity_field_y, shape=(sizeX, sizeY)).numpy()
        Q.set_UVC(u_viz,v_viz)
        plt.savefig(os.path.join(save_path, '{:04d}'.format(t)))

    viz.frames2gif(os.path.join(dir_path, velocity_name), os.path.join(dir_path, velocity_name+".gif"), fps)
    viz.frames2gif(os.path.join(dir_path, density_name), os.path.join(dir_path, density_name+".gif"), fps)
    return trained_vel_x, trained_vel_y