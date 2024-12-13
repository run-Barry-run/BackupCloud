import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import cv2
from multiprocessing import Pool

# Set folder paths
npy_folder_path = '8192_npy'
output_folder = '8192_videos1'
os.makedirs(output_folder, exist_ok=True)

npy_files = [f for f in os.listdir(npy_folder_path) if f.endswith('.npy')]
npy_files.sort()

# Video saving function
def save_video(file_path):
    fps = 1
    frames = fps * 8

    point_cloud = np.load(file_path)
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:] / 255.0  # Normalize RGB values to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    npy_filename = os.path.basename(file_path).replace('.npy', '')
    video_filename = os.path.join(output_folder, f"{npy_filename}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (1024, 1024))

    for frame in range(frames):
        elev_angle = 20 + 10 * np.cos(np.pi * frame / frames)
        azim_angle = 180 * np.sin(2 * np.pi * frame / frames)

        ctr = vis.get_view_control()
        ctr.rotate(azim_angle, elev_angle)

        vis.poll_events()
        vis.update_renderer()

        # Capture image
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(img) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    out.release()
    vis.destroy_window()

def process_file(npy_file):
    file_path = os.path.join(npy_folder_path, npy_file)
    save_video(file_path)

if __name__ == '__main__':
    os.environ["DISPLAY"] = ":0"  # Set the DISPLAY environment variable
    with Pool(1) as pool:
        list(tqdm(pool.imap(process_file, npy_files), total=len(npy_files)))