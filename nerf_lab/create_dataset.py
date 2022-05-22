#!/usr/bin/env python3

import argparse
from omegaconf import OmegaConf
from nerf_lab.matrix import transform_matrix
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

""" Exmaple config:

dataset:
    images: imgs/
    size:
        width: 100
        height: 100
    initial_pose:
        x: 1
        y: 2
        z: 3
        alpha: 0
        beta: 0
        gamma: 0
    transform:
        x: 1
        y: 2
        z: 3
        alpha: 0
        beta: 0
        gamma: 0
    focal_length: 0.001

"""

def main():
    parser = argparse.ArgumentParser("create_dataset")
    parser.add_argument("--config", type=str, help="Config file describing the scene", required=True)
    parser.add_argument("--scene", type=str, help="Path to the created scene", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    pose = transform_matrix(
        alpha=config.dataset.initial_pose.alpha,
        beta=config.dataset.initial_pose.beta,
        gamma=config.dataset.initial_pose.gamma,
        x=config.dataset.initial_pose.x,
        y=config.dataset.initial_pose.y,
        z=config.dataset.initial_pose.z,
    )

    transform = transform_matrix(
        alpha=config.dataset.transform.alpha,
        beta=config.dataset.transform.beta,
        gamma=config.dataset.transform.gamma,
        x=config.dataset.transform.x,
        y=config.dataset.transform.y,
        z=config.dataset.transform.z,
    )

    images_dir = os.path.join(os.path.dirname(args.config), config.dataset.images)
    image_names = os.listdir(images_dir)

    names = { int(name.split('.')[0]):name for name in image_names }
    image_names = [ names[i] for i in sorted(names.keys()) ]
    print(image_names)

    size = (
        config.dataset.size.height,
        config.dataset.size.width
    )

    images = []
    poses = []

    for name in tqdm(image_names):
        path = os.path.join(images_dir, name)
        image = Image.open(path))
        image = image.resize(size)
        image = np.array(image) / 255.
        images.append(np.array(image, dtype=np.float32))
        poses.append(pose.numpy())

        pose = torch.matmul(transform, pose)

    np.savez(args.scene, images=images, poses=poses, focal=np.float32(config.dataset.focal_length))

if __name__ == "__main__":
    main()
