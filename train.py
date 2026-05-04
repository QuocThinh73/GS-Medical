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
from utils.loss_utils import TV_loss, def_reg_loss, l1_loss, ssim, compute_normal_loss, compute_depth_loss, weighted_l1_loss, surface_loss
from gaussian_renderer import render
from gaussian_renderer import network_gui

import time
import sys
from scene import  Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
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
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path

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

        ori_time = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
        d_xyz, d_scales, d_rotations = gaussians.deformation(ori_time)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_scales, d_rotations)
        
        # -----------------------------------------------------------
        # Loss computation
        # -----------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        gt_inpaint_image = viewpoint_cam.inpaint_image.cuda()
        mask = viewpoint_cam.mask.cuda()

        render_inpaint = render_pkg["render_inpaint"]
        render_final = render_pkg["render_final"]
        depth = render_pkg["depth"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # L1_inpaint = l1_loss(render_inpaint, gt_inpaint_image, mask)
        L1_inpaint = weighted_l1_loss(render_inpaint, gt_inpaint_image, 1 + viewpoint_cam.final_conf.cuda(), mask)
        # L1_inpaint = torch.tensor(0.0, device="cuda")
        psnr_inpaint = psnr(render_inpaint, gt_inpaint_image, mask).mean().double()

        # L1_final = l1_loss(render_final, gt_image, mask)
        L1_final = weighted_l1_loss(render_final, gt_image, 1 + viewpoint_cam.final_conf.cuda(), mask)
        psnr_final = psnr(render_final, gt_image, mask).mean().double()

        loss = opt.lambda_final * L1_final + opt.lambda_inpaint * L1_inpaint

        Ll1 = L1_final
        image_for_log = render_final
        gt_for_log = gt_image
        psnr_log = psnr_final

        # depth loss
        if opt.lambda_depth != 0 and viewpoint_cam.original_depth is not None:
            gt_depth = viewpoint_cam.original_depth.cuda().clone()
            pred_depth = depth.clone()

            pred_depth[pred_depth != 0] = 1.0 / pred_depth[pred_depth != 0]
            gt_depth[gt_depth != 0] = 1.0 / gt_depth[gt_depth != 0]

            # L_depth = l1_loss(pred_depth, gt_depth, mask)
            L_depth = weighted_l1_loss(pred_depth, gt_depth, 1.0 - viewpoint_cam.final_conf.cuda(), mask)
            loss += opt.lambda_depth * L_depth
        else:
            L_depth = torch.tensor(0.0, device="cuda")

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
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.7f}",
                    "psnr_inpaint": f"{psnr_inpaint:.2f}",
                    "psnr_final": f"{psnr_final:.2f}",
                    "point": f"{total_point}"
                })
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # Save images
            if args.save_img_from_itr and iteration in args.save_img_from_itr:
                save_example_images(image_for_log, gt_for_log, depth, gt_depth, iteration, dataset.source_path)

            # Log and save
            report_params = {
                "tb_writer": tb_writer,
                "iteration": iteration,
                "Ll1": Ll1,
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
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, 'fine')

            if iteration in args.test_iterations:
                print(
                    f"[ITER {iteration}] "
                    f"Phase: final_training, "
                    f"Total Loss: {loss.item():.7f}, "
                    f"PSNR inpaint: {psnr_inpaint:.2f}, "
                    f"PSNR final: {psnr_final:.2f}, "
                    f"L final: {(opt.lambda_final * L1_final).item():.7f}, "
                    f"L inpaint: {(opt.lambda_inpaint * L1_inpaint).item() if opt.lambda_inpaint != 0 else 0:.7f}, "
                    f"L_depth: {(opt.lambda_depth * L_depth).item() if opt.lambda_depth != 0 else 0:.7f}, "
                )

            if sys_exit:
                sys.exit()

            # -----------------------------------------------------------
            # Adding Gaussian points 
            # -----------------------------------------------------------

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor.grad, visibility_filter)

                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (opt.densify_until_iter)  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*1000 for i in range(0, 100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i*1000 for i in range(0, 100)])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[i*1000 for i in range(0, 100)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_soft_tissues")
    parser.add_argument("--save_img_from_itr", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

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
