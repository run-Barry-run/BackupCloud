import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from multiprocessing import Pool

# Set folder paths
npy_folder_path = '8192_npy'
output_folder = '8192_videos'
os.makedirs(output_folder, exist_ok=True)

npy_files = [f for f in os.listdir(npy_folder_path) if f.endswith('.npy')]
npy_files.sort()

# Function to set equal aspect ratio
def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = x_limits[1] - x_limits[0]
    x_middle = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_middle = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_middle = np.mean(z_limits)
    
    # The plot radius is half of the max range
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Video saving function
def save_video(file_path):
    fps = 1  # 1 frame per second
    total_frames = 8  # Total 8 frames
    
    # Define the standard views
    Views = [
        {'elev':   0, 'azim':   0},   # Front
        {'elev':   0, 'azim': 180},   # Back
        {'elev':   0, 'azim': -90},   # Left
        {'elev':   0, 'azim':  90},   # Right
        {'elev':  90, 'azim':   0},   # Top
        {'elev': -90, 'azim':   0},   # Bottom
        {'elev':  30, 'azim':  45},   # Isometric view 1
        {'elev':  30, 'azim': 135},   # Isometric view 2
    ]
    
    point_cloud = np.load(file_path)
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]

    
    # Create a new figure for each process
    plt.switch_backend('Agg')  # Use a non-GUI backend
    fig = plt.figure(figsize=(8, 8), dpi=128)  # Adjust dpi to get 1024x1024 resolution
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the point cloud
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, marker='o')  # Adjust 's' for point size
    
    # Set axis limits to bounding box
    min_vals = np.min(xyz, axis=0)
    max_vals = np.max(xyz, axis=0)
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    
    set_axes_equal(ax)  # Make sure aspect ratio is equal
    
    ax.set_axis_off()  # Turn off axis
    ax.set_position([0, 0, 1, 1])  # Remove margins
    
    npy_filename = os.path.basename(file_path).replace('.npy', '')
    video_filename = os.path.join(output_folder, f"{npy_filename}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (1024, 1024))
    
    for frame in range(total_frames):
        view = Views[frame % len(Views)]  # Loop back if frames exceed views
        elev_angle = view['elev']
        azim_angle = view['azim']
        ax.view_init(elev=elev_angle, azim=azim_angle)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()
    plt.close(fig)  # Ensure the figure is closed to free memory

def process_file(npy_file):
    file_path = os.path.join(npy_folder_path, npy_file)
    save_video(file_path)

if __name__ == '__main__':
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, npy_files), total=len(npy_files)))
