import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.special import erf
import matplotlib.cm as cm
import tensorflow as tf
import tf_solver_staggered as slv

def frames2gif(src_dir, save_path, fps=30):
    print("Convert frames to gif...")
    filenames = sorted([x for x in os.listdir(src_dir) if x.endswith('.png')])
    img_list = [Image.open(os.path.join(src_dir, name)) for name in filenames]
    img = img_list[0]
    img.save(fp=save_path, append_images=img_list[1:],
            save_all=True, duration=1 / fps * 1000, loop=0)
    print("Done.")

def draw_curl(curl: np.ndarray, save_path=None):
    curl = (curl - np.min(curl))/(np.max(curl) - np.min(curl))
    img = cm.bwr(curl)
    img = Image.fromarray((img * 255).astype('uint8'))
    img_resize = img.resize((640, 640))
    if save_path is not None:
        img_resize.save(save_path)

def draw_density(density: np.ndarray, save_path=None):
    density = erf(np.clip(density, 0, None) * 2)
    img = cm.cividis(density)
    img = Image.fromarray((img * 255).astype('uint8'))
    img_resize = img.resize((640, 640))
    # img_resize = img
    if save_path is not None:
        img_resize.save(save_path)

def draw_velocity(velocity_field_x,velocity_field_y, sizeX, sizeY, coords_x, coords_y, grid_min, grid_step):
    vel_x, vel_y = slv.velocityCentered(velocity_field_x,velocity_field_y, sizeX, sizeY, coords_x, coords_y, grid_min, grid_step)
    u_viz = tf.reshape(vel_x, shape=(sizeX, sizeY)).numpy()
    v_viz = tf.reshape(vel_y, shape=(sizeX, sizeY)).numpy()
    return u_viz, v_viz

def curl(u,v):
    du_dy = np.gradient(u, axis=1)
    dv_dx = np.gradient(v, axis=0)
    return dv_dx - du_dy

def init_viz(u,v,d, coords_x, coords_y, sizeX, sizeY, grid_min, grid_step, v_path, d_path, resolution_limit=30, scale=10):
    x,y = np.meshgrid(coords_x[:sizeX], coords_y[::sizeX])
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    vel_x, vel_y = slv.velocityCentered(u,v, sizeX, sizeY, coords_x, coords_y, grid_min, grid_step)
    Q = ax.quiver(x, y, tf.reshape(vel_x, shape=(sizeX, sizeY)).numpy(), tf.reshape(vel_y, shape=(sizeX, sizeY)).numpy(), color='red', scale=scale, scale_units='width')
    draw_density(np.flipud(tf.reshape(d, shape=(sizeX, sizeX)).numpy()), os.path.join(d_path, '{:04d}.png'.format(0)))
    if sizeX < resolution_limit:
        plt.savefig(os.path.join(v_path, '{:04d}'.format(0)))
    return fig, ax, Q

def init_dir(output_dir, name, size, resolution_limit=30):
    velocity_name = name+"_velocity"
    density_name = name+"_density"
    dir_path = os.path.join(os.getcwd().rsplit("\\",1)[0], output_dir)
    velocity_path = os.path.join(dir_path, velocity_name)
    density_path = os.path.join(dir_path, density_name)
    if size < resolution_limit:
        if not os.path.isdir(velocity_path):
            os.mkdir(velocity_path)
        else:
            for file in os.listdir(velocity_path):
                file_path = os.path.join(velocity_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f'Error deleting file: {file_path} - {e}')
    if not os.path.isdir(density_path):
        os.mkdir(density_path)
    else:
        for file in os.listdir(density_path):
            file_path = os.path.join(density_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Error deleting file: {file_path} - {e}')
    print("Files will be saved here:")
    print(velocity_path)
    print(density_path)
    return velocity_path, density_path, resolution_limit