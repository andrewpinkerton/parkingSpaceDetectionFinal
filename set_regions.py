import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection

# Global variables
points = []
prev_points = []
patches = []
total_points = []
breaker = False
globSelect = None
savePath = "regions.p"  # Default save path

class SelectFromCollection(object):
    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        global points
        points = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

def break_loop(event):
    global breaker, globSelect, savePath
    if event.key == 'b':
        globSelect.disconnect()
        if os.path.exists(savePath):
            os.remove(savePath)

        print("data saved in " + savePath + " file")
        with open(savePath, 'wb') as f:
            pickle.dump(total_points, f, protocol=pickle.HIGHEST_PROTOCOL)
        exit()

def onkeypress(event):
    global points, prev_points, total_points
    if event.key == 'n':
        pts = np.array(points, dtype=np.int32)
        if points != prev_points and len(set(points)) == 4:
            print("Points : " + str(pts))
            patches.append(Polygon(pts))
            total_points.append(pts)
            prev_points = points

def main():
    # Get image path from user
    image_path = input("Enter the path to your image: ")
    output_file = input("Enter the output file name (default: regions.p): ").strip()
    
    global savePath
    if output_file:
        savePath = output_file if output_file.endswith(".p") else output_file + ".p"

    print("\n> Select a region in the figure by enclosing them within a quadrilateral.")
    print("> Press the 'f' key to go full screen.")
    print("> Press the 'esc' key to discard current quadrilateral.")
    print("> Try holding the 'shift' key to move all of the vertices.")
    print("> Try holding the 'ctrl' key to move a single vertex.")
    print("> After marking a quadrilateral press 'n' to save current quadrilateral and then press 'q' to start marking a new quadrilateral")
    print("> When you are done press 'b' to Exit the program\n")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert BGR to RGB for displaying
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Start the annotation loop
    while True:
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)

        # Create a PatchCollection for displaying polygons
        p = PatchCollection(patches, alpha=0.7)
        p.set_array(10 * np.ones(len(patches)))
        ax.add_collection(p)

        # Set up the polygon selector
        global globSelect
        globSelect = SelectFromCollection(ax)

        # Connect event handlers for key presses
        bbox = plt.connect('key_press_event', onkeypress)
        break_event = plt.connect('key_press_event', break_loop)
        
        # Show the plot and wait for user interaction
        plt.show()

        # Disconnect the polygon selector
        globSelect.disconnect()

if __name__ == '__main__':
    main()