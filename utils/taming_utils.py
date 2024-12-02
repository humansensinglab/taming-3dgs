import torch
from PIL import ImageFilter
from gaussian_renderer import render
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms


def get_edges(image):
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    
    return image_edges_tensor

def get_loss_map(reconstructed_image, original_image, config, edges_loss_norm):
    weights = [config["mse_importance"], config["edge_importance"]]
    
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    final_loss = (weights[0] * l1_loss_norm) + \
                (weights[1] * edges_loss_norm)

    return final_loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg, importance_values, opt, to_prune=False):
    config = importance_values

    num_points = len(scene.gaussians.get_xyz)
    gaussian_importance = torch.zeros((len(camlist), num_points), device="cuda", dtype=torch.float32)

    all_opacity = scene.gaussians.get_opacity.detach().squeeze()
    all_scales = torch.prod(scene.gaussians.get_scaling.detach(), dim=1)

    grads = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
    grads[grads.isnan()] = 0.0
    all_grads = grads.detach().squeeze()

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render(my_viewpoint_cam, gaussians, pipe, bg)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        pixel_weights = get_loss_map(render_image, gt_image, config, edge_losses[view].cuda())

        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights = pixel_weights)

        loss_accum = render_pkg["accum_weights"]
        dist_accum = render_pkg["accum_dist"]
        blending_weights = render_pkg["accum_blend"]
        reverse_counts = render_pkg["accum_count"]

        visibility_filter = render_pkg["visibility_filter"].detach()

        all_depths = render_pkg["gaussian_depths"].detach()
        all_radii = render_pkg["gaussian_radii"].detach()

        g_importance = (\
            normalize(config["grad_importance"], all_grads) + \
            normalize(config["opac_importance"], all_opacity) + \
            normalize(config["dept_importance"], all_depths) + \
            normalize(config["radii_importance"], all_radii) + \
            normalize(config["scale_importance"], all_scales) )
        
        p_importance = (
                        normalize(config["dist_importance"], dist_accum) + \
                        normalize(config["loss_importance"], loss_accum) + \
                        normalize(config["count_importance"], reverse_counts) + \
                        normalize(config["blend_importance"], blending_weights)
        )

        agg_importance = config["view_importance"] * photometric_loss * (p_importance + g_importance)
        gaussian_importance[view][visibility_filter] = agg_importance[visibility_filter]
    
    gaussian_importance = gaussian_importance.sum(axis = 0)
    return gaussian_importance


def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def get_count_array(start_count, multiplier, opt, mode):
    # Eq. (2) of taming-3dgs
    if mode == "multiplier":
        budget = int(start_count * float(multiplier))
    elif mode == "final_count":
        budget = multiplier
    
    num_steps = ((opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval)
    slope_lower_bound = (budget - start_count) / num_steps

    k = 2 * slope_lower_bound
    a = (budget - start_count - k*num_steps) / (num_steps*num_steps)
    b = k
    c = start_count

    values = [int(1*a * (x**2) + (b * x) + c) for x in range(num_steps)]

    return values