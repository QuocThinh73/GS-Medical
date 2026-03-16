#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from random import randint
from utils.loss_utils import TV_loss, def_reg_loss, l1_loss, ssim, compute_normal_loss
from gaussian_renderer import render
from gaussian_renderer import network_gui

import time
import sys
from scene import Scene, DeformEnvModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, safe_normalize, reflect, calculate_eccentricities
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.train_utils import prepare_output_and_logger, training_report, setup_seed, save_example_images


def training(dataset, hyper, opt, pipe, args):
    """
    Executes the training loop for the specified dataset and model parameters.

    Parameters:
        dataset (object): The dataset to be used for training.s
        hyper (object): Hyperparameters for flexible deformation modeling.
        opt (object): Optimization parameters.
        pipe (object): Pipeline parameters.
        args (Namespace): Command-line arguments containing various training options.
    """

    # -----------------------------------------------------------
    # Initialize training parameters and setup
    # -----------------------------------------------------------

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_envmap_res, hyper)
    dataset.model_path = args.model_path

    deform_env = DeformEnvModel(t_multires=dataset.t_multires)
    deform_env.train_setting(opt)

    scene = Scene(dataset, gaussians, init_train_args=opt)
    gaussians.training_setup(opt)

    # load checkpoint if specified
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)

    # set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
        
    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------

    print("\n[ITER {}] Saving Checkpoint before training".format(0))
    scene.save(0, 'fine')

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        t_start = time.time()
        # iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        randinteger = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(randinteger)

        # -----------------------------------------------------------
        # Rendering
        # -----------------------------------------------------------

        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        d_reflvec = 0.0
        d_reflvec = torch.tensor(d_reflvec)
        d_reflvec = d_reflvec.to('cuda')
        if iteration < opt.warm_up:
            d_xyz = torch.zeros_like(gaussians.get_xyz)
            d_scales = torch.zeros_like(gaussians._scaling)
            d_rotations = torch.zeros_like(gaussians._rotation)
        else:
            ori_time = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
            if iteration >= opt.warm_up2 and iteration < opt.warm_up3:
                with torch.no_grad():
                    d_xyz, d_scales, d_rotations = gaussians.deformation(ori_time)
            else:
                d_xyz, d_scales, d_rotations = gaussians.deformation(ori_time)

            gb_pos = gaussians.get_xyz + d_xyz 
            view_pos = viewpoint_cam.camera_center.to("cuda").repeat(gaussians.get_opacity.shape[0], 1) 
            d_viewdir_normalized = safe_normalize(view_pos - gb_pos)
            normal, deform_delta_normal = gaussians.get_normal(gaussians.get_scaling, gaussians.get_rotation, d_scales, d_rotations, d_viewdir_normalized) 

        if iteration >= opt.warm_up2:
            gaussians.brdf_mlp.build_mips()
            if iteration >= (opt.warm_up2 + 2000):
                reflvec = safe_normalize(reflect(d_viewdir_normalized, normal))
                ori_time = torch.as_tensor(
                    viewpoint_cam.time,
                    device=reflvec.device,
                    dtype=reflvec.dtype
                ).view(1, 1).expand(reflvec.shape[0], 1)
                d_reflvec = deform_env.step(reflvec.detach(), ori_time)

        gaussians.set_requires_grad("xyz", state=not (iteration >= opt.warm_up2 and iteration < (opt.warm_up3)))
        gaussians.set_requires_grad("opacity", state=not (iteration >= opt.warm_up2 and iteration < (opt.warm_up3)))
        gaussians.set_requires_grad("scaling", state=not (iteration >= opt.warm_up2 and iteration < (opt.warm_up3)))
        gaussians.set_requires_grad("rotation", state=not (iteration >= opt.warm_up2 and iteration < (opt.warm_up3)))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_scales, d_rotations, d_reflvec, iteration, opt)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # -----------------------------------------------------------
        # Loss computation
        # -----------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask.cuda()
        L1_images = l1_loss(image, gt_image, mask)

        loss = (1.0 - opt.lambda_dssim) * L1_images
        psnr_ = psnr(image, gt_image, mask).mean().double()

        normal_loss , delta_normal_loss = 0.0, 0.0
        normal_loss = torch.tensor(normal_loss)
        delta_normal_loss = torch.tensor(delta_normal_loss)
        if iteration >= opt.warm_up + 3000:
            lambda_normal = opt.lambda_normal if iteration >= (opt.warm_up + 3000) else 0.0
            normal_loss = lambda_normal * compute_normal_loss(render_pkg["normal"], render_pkg["normal_ref"])
            
            lambda_delta_normal = calculate_eccentricities(torch.abs(gaussians.get_scaling + d_scales)).unsqueeze(-1) if iteration > (opt.warm_up + 3000) else 0.0
            lambda_delta_normal = lambda_delta_normal if iteration >= (opt.warm_up + 3000) else 0.0
            delta_normal_loss = (lambda_delta_normal * (deform_delta_normal ** 2)).mean()
         
            loss = loss + normal_loss + delta_normal_loss
        
        # if opt.lambda_depth != 0 and viewpoint_cam.original_depth is not None:
        #     gt_depth = viewpoint_cam.original_depth.cuda()

        #     depth[depth!=0] = 1 / depth[depth!=0]
        #     gt_depth[gt_depth!=0] = 1 / gt_depth[gt_depth!=0]

        #     L_depth = l1_loss(depth, gt_depth, mask)
        #     loss += opt.lambda_depth * L_depth

        if opt.lambda_dssim != 0:
            L_dssim = 1.0 - ssim(image, gt_image, mask=mask)
            loss += opt.lambda_dssim * L_dssim

        # if opt.lambda_tv_image != 0:
        #     L_tv_image = TV_loss(image)
        #     loss += opt.lambda_tv_image * L_tv_image

        # if opt.lambda_tv_depth != 0:
        #     L_tv_depth = TV_loss(depth)
        #     loss += opt.lambda_tv_depth * L_tv_depth

        # loss_pos, loss_cov = def_reg_loss(scene.gaussians, d_xyz, d_rotations, d_scales)

        # if opt.lambda_def_reg_pos != 0:
        #     loss += opt.lambda_def_reg_pos * loss_pos

        # if opt.lambda_def_reg_cov != 0:
        #     loss += opt.lambda_def_reg_cov * loss_cov

        sys_exit = False
        if loss.isnan():
            print('nan')
            sys_exit = True

        loss.backward()

        # iter_end.record()

        iter_time = time.time() - t_start

        # -----------------------------------------------------------
        # Training report
        # -----------------------------------------------------------

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Save images
            # if args.save_img_from_itr and iteration in args.save_img_from_itr:
            #     save_example_images(image, gt_image, depth, gt_depth, iteration, dataset.source_path)

            # Log and save
            report_params = {
                "tb_writer": tb_writer,
                "iteration": iteration,
                "Ll1": L1_images,
                "loss": loss,
                "l1_loss": l1_loss,
                "elapsed": iter_time,
                "testing_iterations": args.test_iterations,
                "scene": scene,
                "renderFunc": render,
                "renderArgs": [pipe, background]
            }
            training_report(**report_params)

            if (iteration in args.save_iterations + args.test_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')

            if (iteration in args.test_iterations):
                print(f"[ITER {iteration}] \
                        Total Loss: {loss.item():.{7}f}, \
                        PSNR: {psnr_:.{2}f}, \
                        L1 Loss: {((1.0 - opt.lambda_dssim) * L1_images).item():.{7}f}, \
                        L_dssim: {(opt.lambda_dssim * L_dssim).item() if opt.lambda_dssim != 0 else 0:.{7}f}, \
                        L_normal: {normal_loss}, \
                        L_delta_normal: {delta_normal_loss}")

            if sys_exit:
                sys.exit()
            
            # -----------------------------------------------------------
            # Adding Gaussian points 
            # -----------------------------------------------------------

            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # -----------------------------------------------------------
            # Optimization step
            # -----------------------------------------------------------

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*1000 for i in range(0,100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i*1000 for i in range(0,100)])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[i*1000 for i in range(0,100)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_soft_tissues")
    parser.add_argument("--configs", type=str, default = "arguments/endonerf/default.py")
    parser.add_argument("--save_img_from_itr", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # train
    train_params = {
        'dataset': lp.extract(args),
        'hyper': hp.extract(args),
        'opt': op.extract(args),
        'pipe': pp.extract(args),
        'args': args,
    }
    training(**train_params)

    # Finished
    print("\nTraining complete.")
