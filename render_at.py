#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
import json
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         rendering = render(view, gaussians, pipeline, background)["render"]
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_one(save_path, view, gaussians, pipeline, background):
    rendering = render(view, gaussians, pipeline, background)["render"]
    torchvision.utils.save_image(rendering, save_path)


@torch.no_grad()
def render_at_camera(args, gaussians, background, pipeline: PipelineParams):
    if isinstance(args.camera, str) and args.camera.endswith('.json'):
        with open(args.camera, 'r') as f:
            cam_info = json.load(f)
    else:
        cam_info = args.camera

    W2C_rot = np.array(cam_info["rotation"])
    W2C_pos = np.array(cam_info["position"])
    W2C = np.zeros((4, 4))
    W2C[:3,:3] = W2C_rot
    W2C[:3, 3] = W2C_pos
    W2C[3, 3] = 1.0
    C2W = np.linalg.inv(W2C)

    camera = Camera(
        colmap_id=None,
        R=C2W[:3, :3].transpose(),
        T=C2W[:3, 3],
        FoVx=focal2fov(np.array(cam_info["fx"]), cam_info["intr_width"]),
        FoVy=focal2fov(np.array(cam_info["fy"]), cam_info["intr_height"]),
        image=None,
        gt_alpha_mask=None,
        image_name=None,
        uid=None,
        data_device=args.data_device,
    )
    camera.image_width = cam_info["target_width"]
    camera.image_height = cam_info["target_height"]
    render_one(args.save_path, camera, gaussians, pipeline, background)


def load_gaussians(args, dataset: ModelParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default", rendering_mode="abs")
        gaussians.load_ply(os.path.join(args.gs_path))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return gaussians, background


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--camera", type=str, help="camera info: .json file")
    parser.add_argument("--gs_path", type=str)
    parser.add_argument("--save_path", type=str, default='./tmp/img.png')
    args = parser.parse_args()
    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    model.white_background = False

    gaussians, background = load_gaussians(args, model)

    render_at_camera(args, gaussians, background, pipeline.extract(args))