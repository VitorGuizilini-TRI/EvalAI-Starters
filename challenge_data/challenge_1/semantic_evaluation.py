# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Download data from here:
#   s3://tri-ml-datasets/packnet-sfm/evalai
# Example run:
#   python3 scripts/semantic_evaluation.py --input_folder <path_to_data> --output_folder results --ranges 10 20 50 100 200 --classes All Car Pedestrian Bicycle --metric rmse --use_gt_scale

import argparse
import os
from argparse import Namespace
from collections import OrderedDict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from tqdm import tqdm

map_classes = {
    "Animal": 0,
    "Bicycle": 1,
    "Building": 2,
    "Bus": 3,
    "Car": 4,
    "Caravan/RV": 5,
    "ConstructionVehicle": 6,
    "CrossWalk": 7,
    "Fence": 8,
    "HorizontalPole": 9,
    "LaneMarking": 10,
    "LimitLine": 11,
    "Motorcycle": 12,
    "OtherDrivableSurface": 13,
    "OtherFixedStructure": 14,
    "OtherMovable": 15,
    "Overpass/Bridge/Tunnel": 16,
    "OwnCar(EgoCar)": 17,
    "Pedestrian": 18,
    "Railway": 19,
    "Rider": 20,
    "Road": 21,
    "RoadBarriers": 22,
    "RoadBoundary(Curb)": 23,
    "RoadMarking": 24,
    "SideWalk": 25,
    "Sky": 26,
    "TemporaryConstructionObject": 27,
    "Terrain": 28,
    "TowedObject": 29,
    "TrafficLight": 30,
    "TrafficSign": 31,
    "Train": 32,
    "Truck": 33,
    "Vegetation": 34,
    "VerticalPole": 35,
    "WheeledSlow": 36,
}


def parse_args():
    """Parse arguments for benchmark script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM benchmark script')
    parser.add_argument('--gt_folder', type=str,
                        help='Folder containing predicted depth maps (.npz with key "depth")')
    parser.add_argument('--pred_folder', type=str,
                        help='Folder containing predicted depth maps (.npz with key "depth")')
    parser.add_argument('--output_folder', type=str,
                        help='Output folder where information will be stored')
    parser.add_argument('--use_gt_scale', action='store_true',
                        help='Use ground-truth median scaling on predicted depth maps')
    parser.add_argument('--ranges', type=float, nargs='+', default=[200],
                        help='Depth ranges to consider during evaluation')
    parser.add_argument('--classes', type=str, nargs='+', default=['All', 'Car', 'Pedestrian'],
                        help='Semantic classes to consider during evaluation')
    parser.add_argument('--metric', type=str, default='rmse', choices=['abs_rel', 'rmse', 'silog', 'a1'],
                        help='Which metric will be used for evaluation')
    parser.add_argument('--crop', type=str, default='', choices=['', 'garg'],
                        help='Which crop to use during evaluation')
    parser.add_argument('--min_num_valid_pixels', type=int, default=1,
                        help='Minimum number of valid pixels to consider')
    args = parser.parse_args()
    return args


def create_summary_table(ranges, classes, matrix, folder, metric):

    # Prepare variables
    title = "Semantic/Range Depth Evaluation (%s) -- {}" % metric.upper()
    ranges = ['{}m'.format(r) for r in ranges]
    result = matrix.mean().round(decimals=3)
    matrix = matrix.round(decimals=2)

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(matrix)

    # Show ticks
    ax.set_xticks(np.arange(len(ranges)))
    ax.set_yticks(np.arange(len(classes)))

    # Label ticks
    ax.set_xticklabels(ranges)
    ax.set_yticklabels(classes)

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data to create annotations.
    for i in range(len(ranges)):
        for j in range(len(classes)):
            ax.text(i, j, matrix[j, i],
                    ha="center", va="center", color="w")

    # Plot figure
    ax.set_title(title.format(result))
    fig.tight_layout()

    # Save and show
    plt.savefig('{}/summary_table.png'.format(folder))
    plt.close()


def create_bar_plot(key_range, key_class, matrix, name, idx, folder):

    # Prepare title and start plot
    title = 'Per-frame depth evaluation of **{} at {}m**'.format(key_class, key_range)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get x ticks and values
    x_ticks = [int(m[0]) for m in matrix]
    x_values = range(len(matrix))
    # Get y values
    y_values = [m[2 + idx] for m in matrix]

    # Prepare titles, ticks and labels
    ax.set_title(title)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('Image frame')
    ax.set_ylabel('{}'.format(name.upper()))

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right",
             rotation_mode="anchor")

    # Show and save
    ax.bar(x_values, y_values)
    plt.savefig('{}/{}-{}m-{}.png'.format(folder, key_class, key_range, name))


def load_sem_ins(file):
    """Load GT semantic and instance maps"""
    sem = file.replace('_gt', '_sem')
    if os.path.isfile(sem):
        ins = file.replace('_gt', '_ins')
        sem = imread(sem) / 256.
        ins = imread(ins) / 256.
    else:
        sem = ins = None
    return sem, ins


def load_depth(depth):
    """Load a depth map"""
    depth = imread(depth) / 256.
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
    return depth


def compute_depth_metrics(config, gt, pred, use_gt_scale=True,
                          extra_mask=None, min_num_valid_pixels=1):
    """
    Compute depth metrics from predicted and ground-truth depth maps

    Parameters
    ----------
    config : CfgNode
        Metrics parameters
    gt : torch.Tensor
        Ground-truth depth map [B,1,H,W]
    pred : torch.Tensor
        Predicted depth map [B,1,H,W]
    use_gt_scale : bool
        True if ground-truth median-scaling is to be used
    extra_mask : torch.Tensor
        Extra mask to be used for calculation (e.g. semantic mask)
    min_num_valid_pixels : int
        Minimum number of valid pixels for the image to be considered

    Returns
    -------
    metrics : torch.Tensor [7]
        Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """
    crop = config.crop == 'garg'

    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    abs_diff = abs_rel = sq_rel = rmse = rmse_log = silog = a1 = a2 = a3 = 0.0
    # If using crop
    if crop:
        crop_mask = torch.zeros(gt.shape[-2:]).byte().type_as(gt)
        y1, y2 = int(0.40810811 * gt_height), int(0.99189189 * gt_height)
        x1, x2 = int(0.03594771 * gt_width), int(0.96405229 * gt_width)
        crop_mask[y1:y2, x1:x2] = 1
    # For each depth map
    for pred_i, gt_i in zip(pred, gt):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)

        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > config.min_depth) & (gt_i < config.max_depth)
        valid = valid & crop_mask.bool() if crop else valid
        valid = valid & torch.squeeze(extra_mask) if extra_mask is not None else valid

        # Stop if there are no remaining valid pixels
        if valid.sum() < min_num_valid_pixels:
            return None, None

        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]

        # Ground-truth median scaling if needed
        if use_gt_scale:
            pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)

        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(config.min_depth, config.max_depth)

        # Calculate depth metrics

        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a1 += (thresh < 1.25     ).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel += torch.mean(diff_i ** 2 / gt_i)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(gt_i) -
                                           torch.log(pred_i)) ** 2))

        err = torch.log(pred_i) - torch.log(gt_i)
        silog += torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    # Return average values for each metric
    return torch.tensor([metric / batch_size for metric in
        [abs_rel, sq_rel, rmse, rmse_log, silog, a1, a2, a3]]).type_as(gt), valid.sum()


def main(args):

    # Get and sort ground-truth and predicted files
    pred_files = glob(os.path.join(args.pred_folder, '*.png'))
    pred_files.sort()

    gt_files = glob(os.path.join(args.gt_folder, '*_gt.png'))
    gt_files.sort()

    depth_ranges = args.ranges
    depth_classes = args.classes

    # classes = ['All', 'Car', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Truck', 'Building', 'Road']
    # ranges = [10, 20, 30, 50, 100, 200]

    print('#### Depth ranges to evaluate:', depth_ranges)
    print('#### Depth classes to evaluate:', depth_classes)

    # Metrics name
    metric_names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3']
    matrix_metric = 'rmse'

    # Prepare matrix information
    matrix_idx = metric_names.index(matrix_metric)
    matrix = np.zeros((len(depth_classes), len(depth_ranges)))

    # Create metrics dictionary
    all_metrics = OrderedDict()
    for depth in depth_ranges:
        all_metrics[depth] = OrderedDict()
        for classes in depth_classes:
            all_metrics[depth][classes] = []

    # Loop over all files
    progress_bar = tqdm(zip(pred_files, gt_files), total=len(pred_files))
    for i, (pred_file, gt_file) in enumerate(progress_bar):
        # if i > 20:
        #     break
        # Get and prepare ground-truth and predictions
        gt, pred = load_depth(gt_file), load_depth(pred_file)
        pred = torch.nn.functional.interpolate(pred, gt.shape[2:], mode='nearest')
        # Check for semantics
        sem = gt_file.replace('_gt.png', '_sem.png')
        with_semantic = os.path.exists(sem)
        if with_semantic:
            sem = torch.tensor(load_sem_ins(sem)[0]).unsqueeze(0).unsqueeze(0)
            sem = torch.nn.functional.interpolate(sem, gt.shape[2:], mode='nearest')
        else:
            pass
        # Calculate metrics
        for key_depth in all_metrics.keys():
            for key_class in all_metrics[key_depth].keys():
                # Prepare config dictionary
                args_key = Namespace(**{
                    'min_depth': 0,
                    'max_depth': key_depth,
                    'crop': args.crop,
                })
                # Initialize metrics as None
                metrics, num = None, None
                # Considering all pixels
                if key_class == 'All':
                    metrics, num = compute_depth_metrics(
                        args_key, gt, pred, use_gt_scale=args.use_gt_scale)
                # Considering semantic classes
                elif with_semantic:
                    metrics, num = compute_depth_metrics(
                        args_key, gt, pred, use_gt_scale=args.use_gt_scale,
                        extra_mask=sem == map_classes[key_class],
                        min_num_valid_pixels=args.min_num_valid_pixels)
                # Store metrics if available
                if metrics is not None:
                    metrics = metrics.detach().cpu().numpy()
                    metrics = np.array([i, num] + list(metrics))
                    all_metrics[key_depth][key_class].append(metrics)

    if args.output_folder is None:
        print('#####')
        print(all_metrics)
        print('#####')
        out_dict = {}
        for key1, val1 in all_metrics.items():
            for key2, val2 in val1.items():
                key = '{}_{}m'.format(key2, key1)
                if len(val2) > 0:
                    out_dict[key] = {}
                    for i in range(len(metric_names)):
                        vals = [val2[j][i+2] for j in range(len(val2))]
                        print(i, vals)
                        out_dict[key]['{}'.format(metric_names[i])] = sum(vals) / len(vals)
                else:
                    out_dict[key] = None
        print(out_dict)
        return out_dict

    # Terminal lines
    met_line = '| {:>11} | {:^5} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
    hor_line = '|{:<}|'.format('-' * 109)
    num_line = '| {:>10}m | {:>5} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} |'
    # File lines
    hor_line_file = '|{:<}|'.format('-' * 106)
    met_line_file = '| {:>8} | {:^5} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
    num_line_file = '| {:>8} | {:>5} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} |'
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Loop over the dataset
    for i, key_class in enumerate(depth_classes):
        # Create file and write header
        file = open('{}/{}.txt'.format(args.output_folder, key_class), 'w')
        file.write(hor_line_file + '\n')
        file.write('| ***** {} *****\n'.format(key_class.upper()))
        # Print header
        print(hor_line)
        print(met_line.format(*((key_class.upper()), '#') + tuple(metric_names)))
        print(hor_line)
        # Loop over each depth range and semantic class
        for j, key_depth in enumerate(depth_ranges):
            metrics = all_metrics[key_depth][key_class]
            if len(metrics) > 0:
                # How many metrics were generated for that combination
                length = len(metrics)
                # Update file
                file.write(hor_line_file + '\n')
                file.write(met_line_file.format(*('{}m'.format(key_depth), '#') + tuple(metric_names)) + '\n')
                file.write(hor_line_file + '\n')
                # Create bar plot
                create_bar_plot(key_depth, key_class, metrics, matrix_metric, matrix_idx, args.output_folder)
                # Save individual metric to file
                for metric in metrics:
                    idx, qty, metric = int(metric[0]), int(metric[1]), metric[2:]
                    file.write(num_line_file.format(*(idx, qty) + tuple(metric)) + '\n')
                # Average metrics and update matrix
                metrics = (sum(metrics) / len(metrics))
                matrix[i, j] = metrics[2 + matrix_idx]
                # Print to terminal
                print(num_line.format(*((key_depth, length) + tuple(metrics[2:]))))
                # Update file
                file.write(hor_line_file + '\n')
                file.write(num_line_file.format(*('TOTAL', length) + tuple(metrics[2:])) + '\n')
                file.write(hor_line_file + '\n')
        # Finish file
        file.write(hor_line_file + '\n')
        file.close()
    # Finish terminal printing
    print(hor_line)
    # Create final results
    create_summary_table(depth_ranges, depth_classes, matrix, args.output_folder, args.metric)


if __name__ == '__main__':
    args = parse_args()
    main(args)
