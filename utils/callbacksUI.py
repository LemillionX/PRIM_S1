import os
import sys
import json
import tkinter
import tkinter.filedialog

def prompt_file():
    """Create a Tk file dialog to save file and cleanup when finished"""
    top = tkinter.Tk()
    files = [('JSON file', '*.json')]
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.asksaveasfile(initialdir="../data", parent=top, filetypes=files, defaultextension=files)
    top.destroy()
    if file_name is not None:
        return "../data/"+file_name.name.rsplit("/")[-1]
    return None

def open_file():
    """Create a Tk file dialog to load file and cleanup when finished"""
    top = tkinter.Tk()
    files = [('JSON file', '*.json')]
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.askopenfile(initialdir="../data", parent=top, filetypes=files, defaultextension=files)
    top.destroy()
    return file_name

def points2indices(curves, cell_size, grid_height):
    """
    Gives the grid indices representation of a curve represented by its pixel-points

    Args :
        curves : a list of lists of ``[x,y]`` where ``x,y`` are ``int`` representing the pixel of the curves
        celle_size: an ``int`` representing the size of the grid cells
        grid_height: an ``int`` representing the size of the grid
    
    Returns :
        indices : a list of lists of ``[i,j]`` where ``i,j`` are ``int`` referring to ``cell[i,j]``
    """
    indices = []
    for curve in curves:
        indices.append([])
        for points in curve:
            if len(indices[-1]) == 0 or indices[-1][-1] != [points[0]//cell_size, grid_height-1-points[1]//cell_size]:
                indices[-1].append([points[0]//cell_size, grid_height-1-points[1]//cell_size])
    return indices


def saveToJSON(cells, target_density, init_density, curve, size, file):
    """"
    Parses and saves the data to a ``.json`` file

    Args:
        cells: a list of ``[i,j]`` where ``i,j`` are referring to the cells visited by the trajectory
        target_density: a list of size N representing the target density
        init_density: a list of size N representing the initial density
        curve : a list of of ``[x,y]`` where ``x,y`` are ``int`` representing the pixel of the curve
        size: an ``int`` that is the size of the grid
        file: a ``FileDescriptorOrPath`` where to save the ``.json``
    """
    indices = []
    vel = []
    for i in range(1, len(cells)-1):
        indices.append([cells[i][0] + cells[i][1] * size ])
        vel.append([[cells[i+1][0] - cells[i-1][0]], [cells[i+1][1] - cells[i-1][1]]])

    # Saving indices
    with open(file, 'w') as f:
        json.dump({"indices":indices, "values":vel, "target_density":target_density, "init_density":init_density, "curves": curve}, f, indent=4)

    
def loadFromJSON():
    """
    Loads the data from a ``.json `` file
    """
    file = open_file()
    if file is not None:
        print("Loading {file}".format(file=file))
        data = json.load(file)    
        return data 
    return None