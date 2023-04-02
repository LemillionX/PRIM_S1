import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.special import erf
import matplotlib.cm as cm

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

def tensorToGrid(u, sizeX, sizeY):
    grid = np.zeros((sizeX, sizeY))
    for j in range(sizeY):
        for i in range(sizeX):
            if (i + sizeX*j > sizeX*sizeY-1):
                print(i+sizeX*j)
            grid[j,i] = u[i + sizeX*j]
    return grid

def curl(u,v):
    du_dy = np.gradient(u, axis=1)
    dv_dx = np.gradient(v, axis=0)
    return dv_dx - du_dy


if __name__ in ["__main__", "__builtin__"]:
    OUTPUT_DIR = "output"
    DIR_PATH = os.path.join(os.getcwd().rsplit("\\",1)[0], OUTPUT_DIR)
    DENSITY_NAME = "img"
    target_density =[
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,  
    0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,  
    0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,  
    0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,  
    0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,  
    0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,  
    0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,  
    0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,  
    0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,  
    0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    draw_density(tensorToGrid(target_density, 18, 18), os.path.join(DIR_PATH, DENSITY_NAME, 'target.png'))
