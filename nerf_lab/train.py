import datetime
import os
import typing as t
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import wandb
from loguru import logger

from nerf_lab import data_loading as dl
from nerf_lab import model as mdl
from nerf_lab import render


def run_one_iter_of_tinynerf(
    *,
    tiny_nerf: mdl.TinyNerfModel,
    chunksize: int,
    height: int,
    width: int,
    focal_length: float,
    tform_cam2world: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    depth_samples_per_ray: int,
    encoding_function: t.Callable,
    get_minibatches_function: t.Callable,
) -> torch.Tensor:
    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = render.get_ray_bundle(
        height, width, focal_length, tform_cam2world
    )

    # Sample query points along each ray
    query_points, depth_values = render.compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(tiny_nerf(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render.render_volume_density(
        radiance_field, ray_origins, depth_values
    )

    return rgb_predicted


def train_step(
    *,
    epoch: int,
    log_every_n_step: int,
    test_pose_idx: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    images: torch.Tensor,
    chunksize: int,
    nerf_model: mdl.TinyNerfModel,
    height: int,
    width: int,
    focal_length: float,
    tform_cam2world: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    depth_samples_per_ray: int,
    encoding_function: t.Callable,
    get_minibatches_function: t.Callable,
) -> None:
    target_img_idx = np.random.randint(images.shape[0])
    while target_img_idx == test_pose_idx:
        target_img_idx = np.random.randint(images.shape[0])

    target_img = images[target_img_idx].to(device)
    target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    rgb_predicted = run_one_iter_of_tinynerf(
        tiny_nerf=nerf_model,
        chunksize=chunksize,
        height=height,
        width=width,
        focal_length=focal_length,
        tform_cam2world=target_tform_cam2world,
        near_thresh=near_thresh,
        far_thresh=far_thresh,
        depth_samples_per_ray=depth_samples_per_ray,
        encoding_function=encoding_function,
        get_minibatches_function=get_minibatches_function,
    )

    # Compute mean-squared error between the predicted and target images. Backprop!
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % log_every_n_step == 0:
        psnr = -10.0 * torch.log10(loss)

        wandb.log({"train_loss": loss}, epoch)
        wandb.log({"train_psnr": psnr}, epoch)


def test_step(
    *,
    epoch: int,
    test_img: torch.Tensor,
    tiny_nerf: mdl.TinyNerfModel,
    chunksize: int,
    height: int,
    width: int,
    focal_length: float,
    tform_cam2world: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    depth_samples_per_ray: int,
    encoding_function: t.Callable,
    get_minibatches_function: t.Callable,
) -> t.Tuple[torch.Tensor, float]:
    rgb_predicted = run_one_iter_of_tinynerf(
        tiny_nerf=tiny_nerf,
        chunksize=chunksize,
        height=height,
        width=width,
        focal_length=focal_length,
        tform_cam2world=tform_cam2world,
        near_thresh=near_thresh,
        far_thresh=far_thresh,
        depth_samples_per_ray=depth_samples_per_ray,
        encoding_function=encoding_function,
        get_minibatches_function=get_minibatches_function,
    )

    loss = torch.nn.functional.mse_loss(rgb_predicted, test_img)
    psnr = -10.0 * torch.log10(loss)

    logger.info(f"Test loss {epoch} => {loss.item(): .8f}")
    wandb.log({"test_loss": loss}, epoch)
    wandb.log({"test_psnr": psnr}, epoch)

    return rgb_predicted.detach(), psnr.item()


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="train_experiment"
)
def main(cfg: omegaconf.DictConfig) -> None:
    logger.info("🚀 Training process STARTED!")

    logger.info("🌍 Setup train environment")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        project=cfg.env.wandb_project_name,
        name=f"{cfg.env.wandb_run_name}-{timestamp}",
        entity="wiktorlazarski",
    )
    wandb.config = dict(cfg)

    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda") if cfg.env.with_gpu else torch.device("cpu")

    logger.info("📚 Load scene data")
    data = np.load(cfg.scene.filepath)
    images, tform_cam2world, focal_length = data["images"], data["poses"], data["focal"]

    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
    focal_length = torch.from_numpy(focal_length).to(device)

    # Height and width of each image
    height, width = images.shape[1:3]

    # Hold one image out (for test).
    test_pose_idx = 0
    testimg, testpose = images[test_pose_idx], tform_cam2world[test_pose_idx]
    testimg = torch.from_numpy(testimg).to(device)

    # Map images to device
    images = torch.from_numpy(images[:100, ..., :3]).to(device)

    logger.info("🔫 Create TinyNeRF model")
    nerf_model = mdl.TinyNerfModel(
        hidden_dim=cfg.model.hidden_dim,
        num_encoding_functions=cfg.model.num_encoding_functions,
    )
    nerf_model.to(device)
    wandb.watch(nerf_model)

    logger.info("🏋️‍♂️ Setup training loop")
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=cfg.training.lr)

    psnrs = []
    test_epochs = []
    logger.info("🏋️‍♂️ Training loop")
    for epoch in range(cfg.training.num_epochs):
        train_step(
            epoch=epoch,
            log_every_n_step=cfg.training.train_log_every_n_step,
            test_pose_idx=test_pose_idx,
            optimizer=optimizer,
            device=device,
            images=images,
            chunksize=cfg.scene.chunksize,
            nerf_model=nerf_model,
            height=height,
            width=width,
            focal_length=focal_length,
            tform_cam2world=tform_cam2world,
            near_thresh=cfg.scene.near_thresh,
            far_thresh=cfg.scene.far_thresh,
            depth_samples_per_ray=cfg.training.depth_samples_per_ray,
            encoding_function=lambda x: dl.positional_encoding(
                x, num_encoding_functions=cfg.model.num_encoding_functions
            ),
            get_minibatches_function=dl.get_minibatches,
        )

        if epoch % cfg.training.test_every_n_iterations == 0:
            rendered_img, psnr = test_step(
                epoch=epoch,
                test_img=testimg,
                tiny_nerf=nerf_model,
                chunksize=cfg.scene.chunksize,
                height=height,
                width=width,
                focal_length=focal_length,
                tform_cam2world=testpose,
                near_thresh=cfg.scene.near_thresh,
                far_thresh=cfg.scene.far_thresh,
                depth_samples_per_ray=cfg.training.depth_samples_per_ray,
                encoding_function=lambda x: dl.positional_encoding(
                    x, num_encoding_functions=cfg.model.num_encoding_functions
                ),
                get_minibatches_function=dl.get_minibatches,
            )

            psnrs.append(psnr)
            test_epochs.append(epoch)

            fig, axis = plt.subplots(1, 3, figsize=(12, 4))
            axis[0].imshow(rendered_img.cpu().numpy())
            axis[0].set_title(f"Iteration {epoch}")

            axis[1].plot(test_epochs, psnrs)
            axis[1].set_title("PSNR")

            axis[2].imshow(testimg.detach().cpu().numpy())
            axis[2].set_title("Test scene pose")

            wandb.log({"Test step": fig})
            plt.close("all")

    logger.info("🏁 Training process FINISHED!")

    logger.info("Dumping model")
    torch.save(nerf_model.state_dict(), cfg.training.model)



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
