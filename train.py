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
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss#, fast_ssim
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.taming_utils import compute_gaussian_score, get_edges, get_count_array

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, score_coefficients, args):
    first_iter = 0
    densify_iter_num = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    time_taken = {
            "forward": [],
            "backward": [],
            "step": []
    }
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    all_edges = []
    for view in scene.getTrainCameras():
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())

    counts_array = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    start = time.time()
    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render(cam, gaussians, pipe, background, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()

        if counts_array == None:
            counts_array = get_count_array(len(scene.gaussians.get_xyz), args.budget, opt, mode=args.mode)
            print(counts_array)

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if args.benchmark_dir:
            start_time.record()

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if args.benchmark_dir:
            end_time.record()
            torch.cuda.synchronize()
            time_taken["forward"] += [start_time.elapsed_time(end_time)]

        if args.benchmark_dir:
            start_time.record()
        loss.backward()
        if args.benchmark_dir:
            end_time.record()
            torch.cuda.synchronize()
            time_taken["backward"] += [start_time.elapsed_time(end_time)]

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    edges_stack = all_edges.copy()

                    num_cams = args.cams
                    if args.cams == -1:
                        num_cams = len(my_viewpoint_stack)
                    edge_losses = []
                    camlist = []
                    for _ in range(num_cams):
                        loc = random.randint(0, len(my_viewpoint_stack) - 1)
                        camlist.append(my_viewpoint_stack.pop(loc))
                        edge_losses.append(edges_stack.pop(loc))

                    gaussian_importance = compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg, score_coefficients, opt)                    
                    gaussians.densify_with_score(scores = gaussian_importance, 
                                                max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                budget=counts_array[densify_iter_num+1], 
                                                radii=radii,
                                                iter_num=densify_iter_num)
                    densify_iter_num += 1
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration == args.ho_iteration:
                print("Release opacity limit")
                gaussians.modify_functions()

            # Optimizer step
            if args.benchmark_dir:
                start_time.record()
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if args.sh_lower:
                        if iteration % 16 == 0:
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.shoptimizer.step()
                        gaussians.shoptimizer.zero_grad(set_to_none = True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
            if args.benchmark_dir:
                end_time.record()
                torch.cuda.synchronize()
                time_taken["step"] += [start_time.elapsed_time(end_time)]

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                my_viewpoint_stack = scene.getTrainCameras().copy()

                num_cams = args.cams
                if args.cams == -1:
                    num_cams = len(my_viewpoint_stack)
                camlist = []
                for _ in range(num_cams):
                    camlist.append(my_viewpoint_stack.pop(random.randint(0, len(my_viewpoint_stack) - 1)))

                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if args.benchmark_dir:
            os.makedirs(args.benchmark_dir, exist_ok = True)
            np.save(os.path.join(args.benchmark_dir, "forward.npy"), np.array(time_taken["forward"]))
            np.save(os.path.join(args.benchmark_dir, "backward.npy"), np.array(time_taken["backward"]))
            np.save(os.path.join(args.benchmark_dir, "step.npy"), np.array(time_taken["step"]))

    end = time.time()
    scene.save(iteration)
    print(f"Time taken by {os.getenv('OAR_JOB_ID')}: {end - start}s")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--cams", type=int, default=10)
    parser.add_argument("--budget", type=float, default=20)
    parser.add_argument("--mode", type=str, default="multiplier", choices=["multiplier", "final_count"])
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--ho_iteration", type=int, default=15000)
    parser.add_argument("--sh_lower", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    score_coefficients = {'view_importance': 50, 'edge_importance': 50, 'mse_importance': 50, 'grad_importance': 25, 'dist_importance': 50, 'opac_importance': 100, 'dept_importance': 5, 'loss_importance': 10, 'radii_importance': 10, 'scale_importance': 25, 'count_importance': 0.1, 'blend_importance': 50}
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets, 
        score_coefficients, 
        args
    )

    # All done
    print("\nTraining complete.")
