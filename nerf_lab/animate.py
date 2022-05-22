#!/usr/bin/env python3

import argparse
from nerf_lab.train import run_one_iter_of_tinynerf
from nerf_lab.model import TinyNerfModel
from nerf_lab.data_loading import positional_encoding, get_minibatches
from nerf_lab.matrix import rotation_matrix
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from loguru import logger

def main():
    parser = argparse.ArgumentParser("animate")
    parser.add_argument("--width", type=int, help="Rendered image width", default=100)
    parser.add_argument("--height", type=int, help="Rendered image height", default=100)
    parser.add_argument("--model", type=str, help="Path to saved model", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_pose_idx", type=int, help="Starting pose", default=0)
    parser.add_argument("--alpha_rotation_angle", type=float, help="Rotation angle in degrees", default=0)
    parser.add_argument("--beta_rotation_angle", type=float, help="Rotation angle in degrees", default=0)
    parser.add_argument("--gamma_rotation_angle", type=float, help="Rotation angle in degrees", default=0)
    parser.add_argument("--config", type=str, help="Path to config file", required=True)

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    data = np.load(config.scene.filepath)
    tform_cam2world, focal_length = data["poses"], data["focal"]

    device = torch.device(args.device)
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
    focal_length = torch.from_numpy(focal_length).to(device)
    pose = tform_cam2world[args.start_pose_idx]

    nerf_model = TinyNerfModel(
        hidden_dim=config.model.hidden_dim,
        num_encoding_functions=config.model.num_encoding_functions
    )

    nerf_model.load_state_dict(torch.load(args.model))
    nerf_model.to(device)

    encoding_function = partial(positional_encoding,
         num_encoding_functions=config.model.num_encoding_functions)

    frame = 0
    plt.ion()
    plt.show()

    alpha = args.alpha_rotation_angle / 360 * 2 * 3.1415
    beta = args.beta_rotation_angle / 360 * 2 * 3.1415
    gamma = args.gamma_rotation_angle / 360 * 2 * 3.1415
    transform_matrix = rotation_matrix(alpha=alpha, beta=beta, gamma=gamma).to(device)

    while True:
        try:
            rgb_predicted = run_one_iter_of_tinynerf(
                tiny_nerf=nerf_model,
                chunksize=config.scene.chunksize,
                height=args.height,
                width=args.width,
                focal_length=focal_length,
                tform_cam2world=pose,
                near_thresh=config.scene.near_thresh,
                far_thresh=config.scene.far_thresh,
                depth_samples_per_ray=config.training.depth_samples_per_ray,
                encoding_function=encoding_function,
                get_minibatches_function=get_minibatches,
            )

            logger.info(f"Drawing {frame} frame")
            frame += 1
            img = rgb_predicted.detach().cpu().numpy()
            plt.imshow(img)
            plt.title(f'Frame {frame}')
            plt.draw()
            plt.pause(.001)

            del rgb_predicted
            pose = torch.matmul(transform_matrix, pose)

        except KeyboardInterrupt:
            break

    logger.info("Shutting down")






if __name__ == "__main__":
    main()
