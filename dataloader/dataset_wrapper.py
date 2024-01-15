
import numpy as np
import torch
import numba as nb
from torch.utils import data
from dataloader.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage
from typing import List, Union
from torch import Tensor
import torch.nn.functional as F

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

def voxel_coords_to_world_coords(
    aabb_min: Union[Tensor, List[float]],
    aabb_max: Union[Tensor, List[float]],
    voxel_resolution: Union[Tensor, List[int]],
    points: Union[Tensor, List[float]] = None,
) -> Tensor:
    aabb_min = torch.tensor(aabb_min) if isinstance(aabb_min, List) else aabb_min
    aabb_max = torch.tensor(aabb_max) if isinstance(aabb_max, List) else aabb_max
    voxel_resolution = (
        torch.tensor(voxel_resolution)
        if isinstance(voxel_resolution, List)
        else voxel_resolution
    )

    if points is None:
        x, y, z = torch.meshgrid(
            torch.linspace(aabb_min[0], aabb_max[0], voxel_resolution[0]),
            torch.linspace(aabb_min[1], aabb_max[1], voxel_resolution[1]),
            torch.linspace(aabb_min[2], aabb_max[2], voxel_resolution[2]),
        )
        return torch.stack([x, y, z], dim=-1)
    else:
        points = torch.tensor(points) if isinstance(points, List) else points

        # Compute voxel size
        voxel_size = (aabb_max - aabb_min) / voxel_resolution

        # Convert voxel coordinates to world coordinates
        world_coords = aabb_min.to(points.device) + points * voxel_size.to(points.device)

        return world_coords

def world_coords_to_voxel_coords(
    point: Union[Tensor, List[float]],
    aabb_min: Union[Tensor, List[float]],
    aabb_max: Union[Tensor, List[float]],
    voxel_resolution: Union[Tensor, List[int]],
) -> Tensor:
    # Convert lists to tensors if necessary
    point = torch.tensor(point) if isinstance(point, List) else point
    aabb_min = torch.tensor(aabb_min) if isinstance(aabb_min, List) else aabb_min
    aabb_max = torch.tensor(aabb_max) if isinstance(aabb_max, List) else aabb_max
    voxel_resolution = (
        torch.tensor(voxel_resolution)
        if isinstance(voxel_resolution, List)
        else voxel_resolution
    )

    # Compute the size of each voxel
    voxel_size = (aabb_max - aabb_min) / voxel_resolution

    # Compute the voxel index for the given point
    voxel_idx = ((point - aabb_min) / voxel_size).long()

    return voxel_idx

class DatasetWrapper_NuScenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, fill_label=0,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3], 
                 min_volume_space=[-51.2, -51.2, -5], phase='train', scale_rate=1):
        'Initialization'
        self.imagepoint_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        if scale_rate != 1:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
        else:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
        self.transforms = transforms

    def __len__(self):
        return len(self.imagepoint_dataset)

    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]
        imgs, img_metas, xyz, labels = data

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        # TODO: intervals should not minus one.
        intervals = crop_range / (cur_grid_size - 1)   

        if (intervals == 0).any(): 
            print("Zero interval!")
        # TODO: grid_ind_float should actually be returned.
        # grid_ind_float = (np.clip(xyz, min_bound, max_bound - 1e-3) - min_bound) / intervals
        grid_ind_float = (np.clip(xyz, min_bound, max_bound) - min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (imgs, img_metas, processed_label)

        data_tuple += (grid_ind, labels)

        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def custom_collate_fn(data):
    img2stack = np.stack([d[0] for d in data]).astype(np.float32)
    meta2stack = [d[1] for d in data]
    label2stack = np.stack([d[2] for d in data]).astype(np.int)
    # because we use a batch size of 1, so we can stack these tensor together.
    grid_ind_stack = np.stack([d[3] for d in data]).astype(np.float)
    point_label = np.stack([d[4] for d in data]).astype(np.int)
    return torch.from_numpy(img2stack), \
        meta2stack, \
        torch.from_numpy(label2stack), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label)
        
class DatasetWrapper_NuScenes_NeRF(data.Dataset):
    def __init__(self, in_dataset, grid_size, fill_label=0,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3], 
                 min_volume_space=[-51.2, -51.2, -5], phase='train', scale_rate=1):
        'Initialization'
        self.imagepoint_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        if scale_rate != 1:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
        else:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
        self.transforms = transforms

    def __len__(self):
        return len(self.imagepoint_dataset)

    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]
        imgs, img_metas, nerf_dino_feat, nerf_voxel_coords = data

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        assert self.fixed_volume_space
        max_bound = torch.Tensor(self.max_volume_space)  # 51.2 51.2 3
        min_bound = torch.Tensor(self.min_volume_space)  # -51.2 -51.2 -5
        
        # add by Yuxi Qian
        nerf_voxel_resolution_scene = img_metas['nerf_voxel_resolution']
        nerf_aabb_min_scene = torch.Tensor(img_metas['nerf_aabb_min'])
        nerf_aabb_max_scene = torch.Tensor(img_metas['nerf_aabb_max'])
        lidar_to_first_nerf_ego_transform = torch.Tensor(img_metas['lidar_to_first_ego_transform'])
        
        nerf_voxel_coords_world = voxel_coords_to_world_coords(nerf_aabb_min_scene, nerf_aabb_max_scene, nerf_voxel_resolution_scene, nerf_voxel_coords)
        nerf_world_coords_homo = torch.cat((nerf_voxel_coords_world, torch.ones(nerf_voxel_coords_world.shape[0], 1)), dim=1)
        nerf_world_coords_extended_lidar_t_sample = (nerf_world_coords_homo @ torch.inverse(lidar_to_first_nerf_ego_transform).T)[:,:3]
        
        # Create a mask for each condition
        mask_x = (nerf_world_coords_extended_lidar_t_sample[:, 0] >= min_bound[0]) & (nerf_world_coords_extended_lidar_t_sample[:, 0] <= max_bound[0])
        mask_y = (nerf_world_coords_extended_lidar_t_sample[:, 1] >= min_bound[1]) & (nerf_world_coords_extended_lidar_t_sample[:, 1] <= max_bound[1])
        mask_z = (nerf_world_coords_extended_lidar_t_sample[:, 2] >= min_bound[2]) & (nerf_world_coords_extended_lidar_t_sample[:, 2] <= max_bound[2])

        # Combine masks to identify rows that meet all conditions
        mask = mask_x & mask_y & mask_z

        # Apply the mask to filter the tensors
        filtered_nerf_sample_world_coords = nerf_world_coords_extended_lidar_t_sample[mask]
        filtered_nerf_sample_feats = nerf_dino_feat[mask]
        
        filtered_nerf_sample_vox_coords = world_coords_to_voxel_coords(filtered_nerf_sample_world_coords, min_bound, max_bound, self.grid_size)
        filtered_nerf_voxels_dino_feature_sample = torch.zeros((*self.grid_size, 64))
        filtered_nerf_voxels_dino_feature_sample[filtered_nerf_sample_world_coords[...,0].long(),filtered_nerf_sample_world_coords[...,1].long(),filtered_nerf_sample_world_coords[...,2].long()] = filtered_nerf_sample_feats
        filtered_nerf_voxels_dino_feature_sample = filtered_nerf_voxels_dino_feature_sample.permute(3,0,1,2) # (embed_dim, x, y, z)
        
        data_tuple = (imgs, img_metas, filtered_nerf_voxels_dino_feature_sample)
        
        return data_tuple



def custom_collate_fn_w_nerf(data):
    img2stack = np.stack([d[0] for d in data]).astype(np.float32)
    meta2stack = [d[1] for d in data]
    nerf_feat2stack = torch.stack([d[2] for d in data]).float()
    return torch.from_numpy(img2stack), \
        meta2stack, \
        nerf_feat2stack