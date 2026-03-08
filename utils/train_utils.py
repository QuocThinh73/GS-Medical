
import os
import numpy as np
import torch
import random
from argparse import Namespace
from scene import Scene
from PIL import Image


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):
    """
    Prepares the output folder and Tensorboard logger.

    Parameters:
        args (Namespace): Arguments from the command line.
    """ 
    if not args.model_path:
        unique_str = args.expname
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
        # log basic metrics
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)

def save_image(tensor, filename, source_path):
    """
    Saves a tensor as an image.

    Parameters:
        tensor (Tensor): Tensor to save as an image.
        filename (str): Name of the file to save the image to.
        source_path (str): Path to the folder where the image should be saved.
    """
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 1: 
        array = np.squeeze(array, axis=0)
    else:
        array = array.transpose(1, 2, 0)  # Convert from CHW to HWC for RGB images.
    array = (array * 255).astype(np.uint8)
    Image.fromarray(array).save(os.path.join(source_path, filename))

def save_example_images(image, gt_image, depth, gt_depth, iteration, source_path):
    """
    Saves example images for debugging purposes.
    
    Parameters:
        image (Tensor): Rendered image.
        gt_image (Tensor): Ground truth image.
        depth (Tensor): Rendered depth map.
        gt_depth (Tensor): Ground truth depth map.
        iteration (int): Current iteration number.
        source_path (str): Path to the folder where the images should be saved.
    """
    save_image(image, "render_" + str(iteration) + ".png", source_path)
    save_image(gt_image, "gt_" + str(iteration) + ".png", source_path)
    save_image(depth, "depth_" + str(iteration) + ".png", source_path)
    save_image(gt_depth, "gt_depth_" + str(iteration) + ".png", source_path)