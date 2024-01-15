import os
import pickle
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
import json
import time

def transform_matrix(translation, rotation):
    """Compute transformation matrix given translation and rotation (in quaternion)."""
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = translation
    return transform

def get_global_to_ego_matrix(ego_pose):
    ego_to_global = np.eye(4)
    ego_to_global[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
    ego_to_global[:3, 3] = ego_pose["translation"]
    return np.linalg.inv(ego_to_global)
        
def get_transformation_to_first_ego(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']

    # Get the first sample of the scene
    scene = nusc.get('scene', scene_token)
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    current_camera_data_token = first_sample["data"]["CAM_FRONT"]
    camera_data_dict = nusc.get("sample_data", current_camera_data_token)
    initial_ego_pose = nusc.get("ego_pose", camera_data_dict["ego_pose_token"])
    global_to_initial_ego_matrix = get_global_to_ego_matrix(initial_ego_pose)

    # Get the lidar_top data
    current_lidar_data_token = sample["data"]["LIDAR_TOP"]
    current_lidar_data = nusc.get("sample_data", current_lidar_data_token)
    # 1. lidar --> current ego pose
    lidar_intrinsic = nusc.get(
        "calibrated_sensor", current_lidar_data["calibrated_sensor_token"]
    )
    lidar_to_ego = np.eye(4)
    lidar_to_ego[:3, :3] = Quaternion(
        lidar_intrinsic["rotation"]
    ).rotation_matrix
    lidar_to_ego[:3, 3] = np.array(lidar_intrinsic["translation"])
    # 2. current ego pose --> global pose
    # Construct the transformation matrix for this timestamp
    ego_pose = nusc.get("ego_pose", current_lidar_data["ego_pose_token"])
    ego_to_global_matrix = np.eye(4)
    ego_to_global_matrix[:3, :3] = Quaternion(
        ego_pose["rotation"]
    ).rotation_matrix
    ego_to_global_matrix[:3, 3] = np.array(ego_pose["translation"])
    # 3. global pose --> ego pose at t=0
    return global_to_initial_ego_matrix @ ego_to_global_matrix @ lidar_to_ego

def main(pkl_path, nusc):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    print(f"Loaded {len(pkl_data['infos'])} samples from the pickle file.")

    pkl_infos = pkl_data['infos']
    for index, pkl_sample in tqdm(enumerate(pkl_infos), total=len(pkl_infos)):
        nusc_sample_token = pkl_sample['token']
        nusc_sample = nusc.get('sample', nusc_sample_token)
        nusc_scene_token = nusc_sample['scene_token']
        
        lidar_to_first_ego_transform = get_transformation_to_first_ego(nusc, nusc_sample['token'])
        #read the scene nerf meta info
        base_nerf_save_dir = "/lpai/volumes/perception/qyx/nusc_nerf/dino_feat"
        base_nerf_save_scene_dir = os.path.join(base_nerf_save_dir, nusc_scene_token)
        
        dino_feat_path = os.path.join(base_nerf_save_scene_dir, "dino_feature.npy")
        voxel_coords_path = os.path.join(base_nerf_save_scene_dir, "voxel_coords.npy")
        nerf_metadata_path = os.path.join(base_nerf_save_scene_dir, "metadata.json")
        
        with open(nerf_metadata_path, 'r') as file:
            nerf_metadata = json.load(file)
        
        pkl_sample['scene_token'] = nusc_scene_token
        pkl_sample['lidar_to_first_ego_transform'] = lidar_to_first_ego_transform
        pkl_sample['nerf_dino_feat_path'] = dino_feat_path   
        pkl_sample['nerf_voxel_coords_path'] = voxel_coords_path
        pkl_sample['nerf_aabb_min'] = nerf_metadata['aabb_min']
        pkl_sample['nerf_aabb_max'] = nerf_metadata['aabb_max']
        pkl_sample['nerf_voxel_size'] = nerf_metadata['voxel_size']
        pkl_sample['nerf_voxel_resolution'] = nerf_metadata['voxel_resolution']
        pkl_sample['nerf_density_threshold'] = nerf_metadata['density_threshold']
        
    # save the augmented pickle file, rename is orginal name + "_aug.pkl"
    start_time_save = time.time()
    pkl_save_path = pkl_path[:-4] + "_aug.pkl"
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        
    print(f"Saved {len(pkl_data['infos'])} samples to the pickle file. Time cost: {time.time() - start_time_save} seconds.")
        
if __name__ == '__main__':
    # Initialize the NuScenes object
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/jovyan/TPVFormer/data/nuscenes', verbose=True)
    print("Loaded NuScenes Database")
    main("/home/jovyan/TPVFormer/data/nuscenes_infos_train.pkl", nusc)
    main("/home/jovyan/TPVFormer/data/nuscenes_infos_val.pkl", nusc)


        
        