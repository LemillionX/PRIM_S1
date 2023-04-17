import os
import sys
import json
import tkinter
import tkinter.filedialog


def prompt_file():
    """Create a Tk file dialog and cleanup when finished"""
    top = tkinter.Tk()
    files = [('JSON file', '*.json')]
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.asksaveasfile(initialdir="../data", parent=top, filetypes=files, defaultextension=files)
    top.destroy()
    return "../data/"+file_name.name.rsplit("/")[-1]

def saveToJSON(cells,size, file):
    indices = []
    vel = []
    for i in range(1, len(cells)-1):
        indices.append([cells[i][0] + cells[i][1] * size ])
        vel.append([[cells[i+1][0] - cells[i-1][0]], [cells[i+1][1] - cells[i-1][1]]])

    # Saving indices
    with open(file, 'w') as f:
        json.dump({"indices":indices, "values":vel}, f)

    
    

