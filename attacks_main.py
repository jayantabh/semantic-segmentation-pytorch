"""Generates adversarial examples for segmentation and image classification models."""

# System libs
import os
import argparse
from distutils.version import LooseVersion
import sys
import datetime
import time
import shutil
import csv

# Numerical libs
import torch
import torch.nn as nn
import cv2
import numpy as np
import scipy.io as sio
from scipy.io import loadmat

# Our libs
from dataset import ValDataset, AttackDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger, find_recursive, create_dir
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from lib.attacks import attacks
from PIL import Image
from tqdm import tqdm
from config import cfg
import matplotlib.pyplot as plt


__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'


adv_attacks = {
    'fgsm': attacks.fgsm,
    'targetted_fgsm': attacks.fgsm_targetted,
    'iterative_fgsm': attacks.IterativeFGSM,
    'iterative_fgsm_ll': attacks.IterativeFGSMLeastLikely
}

colors = loadmat('data/color150.mat')['colors']


def encode_save_image(arr, path):
    # encode array
    img_color = colorEncode(arr, colors)

    Image.fromarray(img_color).save(path)


def evaluate(segmentation_module, loader, cfg, task, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    save_pred =  cfg.ATTACK[task + '_save_pred']
    save_image = cfg.ATTACK[task + '_save_pred_img']

    preds = []
    # confs = []
    iou_map = dict()
    pred_path = None

    assert task in ['orig', 'pert'], "task should be pred or orig for attack evaluation"

    if save_pred or save_image:
        pred_path = os.path.join(cfg.ATTACK.output_dir, 'pred')

        if not os.path.isdir(pred_path):
            create_dir(pred_path)

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

            preds.append(pred)
            # TODO: See if required
            # confs.append(scores.detach().numpy())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)

        iou_map[batch_data['info'].split('/')[-1].replace('.jpg', '')] = intersection.sum() / union.sum()

        intersection_meter.update(intersection)
        union_meter.update(union)

        # save prediction
        if save_pred:
            img_name = batch_data['info'].split('/')[-1]
            np.save(os.path.join(pred_path, task + '_pred_' + img_name.replace('.jpg', '.npy')), np.array(pred))

        # save predicted image
        if save_image:
            img_name = batch_data['info'].split('/')[-1]
            image_path = os.path.join(pred_path, task + '_image_' + img_name.replace('.jpg', '.png'))
            encode_save_image(pred, image_path)

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))

    results = dict(
        iou=iou_map,
        class_iou=iou,
        accuracy=acc_meter.average()*100
    )

    return np.array(preds), None, results


# def Predict(net, x, dummy_label=None, label_names=None, do_top_5=False, is_seg=False):
#     """Performs a forward pass of the image.
#
#        net: The caffe network. Assumes that the network definition has the following keys:
#         "data" - input image
#         "label" - the label used to compute the loss.
#         "output" - predicted logits by the network.
#        x: The data to be passed through the network
#        dummy_label: Model definitions for adversarial examples have a label input as well
#                     for computing the loss. For only prediction, this can be set arbitrarily
#        label_names: The names of each class
#        do_top_5: For image classification models, whether to show top 5 predictions
#        is_seg: Whether 'net' is a segmentation model (true) or image classification model (false)
#     """
#
#     net.blobs['data'].data[0,:,:,:] = np.squeeze(x)
#
#     if dummy_label is None:
#         net.blobs['label'].data[...] = np.zeros( net.blobs['label'].data.shape )
#     else:
#         net.blobs['label'].data[...] = dummy_label
#
#     net.forward()
#
#     net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
#     confidence = net.blobs['output'].data[0].astype(np.float32)
#
#     if is_seg:
#         return net_prediction, confidence
#
#     return net_prediction, confidence, pred_label


def get_adv_func_args(cfg, model, loader, gpu):
    """Prepares keyword arguments for adversarial attack functions."""

    attack_method = cfg.ATTACK.method.lower()
    adv_args = {}

    if attack_method == 'fgsm':
        adv_args = {
            'model': model,
            'loader': loader,
            'eps': cfg.ATTACK.eps,
            'out_dir': 'pert',
            'gpu': gpu
        }
    elif attack_method == 'targetted_fgsm':
        adv_args = {
            'model': model,
            'loader': loader,
            'eps': cfg.ATTACK.eps,
            'target_idx': cfg.ATTACK.target_idx,
            'out_dir': os.path.join(cfg.DATASET.root_dataset, 'pert'),
            'gpu': gpu
        }
    elif attack_method in ['iterative_fgsm', 'iterative_fgsm_ll']:
        adv_args = {
            'model': model,
            'loader': loader,
            'eps': cfg.ATTACK.eps,
            'num_iters': cfg.ATTACK.num_iters,
            'alpha': cfg.ATTACK.alpha,
            'do_stop_max_pert': cfg.ATTACK.do_max_pert,
            'out_dir': os.path.join(cfg.DATASET.root_dataset, 'pert'),
            'gpu': gpu
        }
    else:
        raise AssertionError('Unknown attack method')

    return adv_args


# def PredictWrapper(net, image, orig_image, dummy_label=None, is_seg=True, args=None):
#     """Wrapper for calling the Predict function. DilatedNet has its own pre- and post-processing."""
#
#     if not args.is_dilated:
#         pred, conf = Predict(net, image, dummy_label=dummy_label, is_seg=is_seg)
#     else:
#         _, conf = Predict(net, image, dummy_label=dummy_label, is_seg=is_seg)
#         pred = dilated.PostprocessPrediction(conf, orig_image, args.dataset)
#         pred = pred.argmax(axis=0).astype(np.uint32)
#
#     pred = pred[0:orig_image.shape[0], 0:orig_image.shape[1]]
#     conf = conf[:, 0:orig_image.shape[0], 0:orig_image.shape[1]]
#     return pred, conf


def check_already_processed(args, image_name, model_name):
    """Checks if the image has already been saved in the output directory."""

    if args.force_overwrite:
        return False

    if args.save_adv_example:
        output_template = "{}_advinput_{}_eps={}_target_idx={}.mat"
    else:
        output_template = "{}_pert_{}_eps={}_target_idx={}.png"
    output_name = output_template.format(
        image_name, model_name, args.eps, args.target_idx)

    if os.path.exists(os.path.join(args.out_dir, output_name)):
        return True

    return False


def check_all_done(args):
    """Checks if all images have already been saved in the output directory."""

    image_names = open(args.image_file, 'r').readlines()
    image_names = [x.strip() for x in image_names]

    for im_path in image_names:
        im_name = os.path.basename(im_path).split('.')[0].replace('_leftImg8bit', '')

        is_done = check_already_processed(args, im_name, args.model_name)
        if not is_done:
            return False

    return True

# def main_image(args):
#     """Adversarial example for ImageNet classification model."""
#
#     if args.gpu >= 0:
#         caffe.set_mode_gpu()
#         caffe.set_device(args.gpu)
#
#     net = caffe.Net(args.model_def, args.model_weights, caffe.TEST)
#
#     transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#     transformer.set_mean('data', args.mean)
#     transformer.set_transpose('data', (2,0,1))
#     transformer.set_channel_swap('data', (2,1,0))
#     transformer.set_raw_scale('data', 255.0)
#
#     image = caffe.io.load_image(args.image)
#     image = transformer.preprocess('data', image)
#
#     print("Prediction of original image")
#     Predict(net, image, do_top_5=True, label_names=args.label_names)
#
#     adv_func_args = GetAdvFuncArgs(args, net, image)
#     adversarial_image_data, added_noise_data = adv_attacks[args.attack_method](**adv_func_args)
#
#     print("Prediction of adversarial image")
#     Predict(net, adversarial_image_data, do_top_5=True, label_names=args.label_names)
#
#     adversarial_image = np.squeeze(adversarial_image_data[0,:,:,:]) # CxHxW
#     adversarial_image = np.transpose( adversarial_image, [1,2,0] ) # HxWxC
#     cv2.imwrite("adversarial_example.png", adversarial_image + args.mean)
#
#     added_noise = np.transpose( np.squeeze(added_noise_data[0,:,:,:]), [1,2,0] )
#     cv2.imwrite("perturbation.png", added_noise + args.mean)


def main(cfg, gpu):
    """Adversarial examples for semantic segmentation models evaluated over all images in a list file."""

    # if check_all_done(cfg):
    #     print("Processing", cfg.ATTACK.list_attack)
    #     print("Arguments", sys.argv)
    #     print("Entire experiment is already done. Quitting")
    #     return

    # Save the command line args used to run the program
    f_cmdline = open(os.path.join(cfg.ATTACK.output_dir,
                                  'cmdline' + str(datetime.datetime.now()).replace(' ', '_') + '.txt'), 'w')
    for i in range(0, len(sys.argv)):
        f_cmdline.write(sys.argv[i] + " ")
    f_cmdline.close()

    torch.cuda.device(gpu)

    cfg.MODEL.model_name = os.path.basename(cfg.MODEL.arch_encoder + '_' + cfg.MODEL.arch_decoder)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)

    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    # Copy the model prototxt to the folder
    # shutil.copyfile(args.model_def, os.path.join(args.out_dir, model_name + '.prototxt'))

    # Create the network and start the actual experiment
    criterion = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, criterion)

    # image_names = open(args.image_file, 'r').readlines()
    # image_names = [x.strip() for x in image_names]
    #
    # for i, im_name in enumerate(image_names):
    #     args.image = im_name
    #     main_seg(args, net)
    #
    #     if i % args.iter_print == 0:
    #         time_str = str(datetime.datetime.now())
    #         print("[{}] Image {}: {}".format(time_str, i, im_name))
    #         sys.stdout.flush()

    # Dataset and Loader
    orig_dataset = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.ATTACK.list_attack,
        cfg.DATASET)

    orig_loader = torch.utils.data.DataLoader(
        orig_dataset,
        batch_size=cfg.ATTACK.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # image_name = os.path.basename(args.image).split('.')[0].replace('_leftImg8bit', '')

    # TODO: Check if required and modify accordingly
    # if check_already_processed(cfg, image_name, model_name):
    #     return

    # if net is None:
    #     net = caffe.Net(args.model_def, args.model_weights, caffe.TEST)

    # image, im_height, im_width, orig_image = lib.PreprocessImage(
    #     args.image, args.pad_size, pad_value=args.pad_value, resize_dims=args.resize_dims, args=args)

    # orig_pred, orig_conf = PredictWrapper(
    #     net, image, orig_image, dummy_label=None, is_seg=True, args=args)

    orig_pred, _, orig_results = evaluate(segmentation_module,
                                                  orig_loader,
                                                  cfg, 'orig', gpu)

    attack_dataset = AttackDataset(
        cfg.DATASET.root_dataset,
        cfg.ATTACK.list_attack,
        cfg.DATASET)

    adv_func_args = get_adv_func_args(cfg, segmentation_module, attack_dataset, gpu)
    list_adv, _ = adv_attacks[cfg.ATTACK.method](**adv_func_args)

    pert_dataset = ValDataset(
        cfg.DATASET.root_dataset,
        list_adv,
        cfg.DATASET)

    pert_loader = torch.utils.data.DataLoader(
        pert_dataset,
        batch_size=cfg.ATTACK.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    adv_pred, _, adv_results = evaluate(segmentation_module,
                                        pert_loader,
                                        cfg, 'pert', gpu)

    print('Segmentation attack processing complete!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        type=str,
        help="an image paths, or a directory name containing images on which to perform attack"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exists!"

    assert cfg.ATTACK.use_val or args.imgs is not None, \
        "One out of use validation flag or imgs should be provided"

    # generate testing image list
    if cfg.ATTACK.use_val:
        cfg.ATTACK.list_attack = cfg.DATASET.list_val
    else:
        if os.path.isdir(args.imgs[0]):
            imgs = find_recursive(args.imgs[0])
        else:
            imgs = [args.imgs]
        cfg.ATTACK.list_attack = [{'fpath_img': x.strip()} for x in imgs]

    assert cfg.ATTACK.use_val or len(imgs), "either use_val should be true or " + \
                                            "imgs should be a path to image (.jpg) or directory."

    if not os.path.isdir(cfg.ATTACK.output_dir):
        create_dir(cfg.ATTACK.output_dir)

    if not os.path.isdir(os.path.join(cfg.DATASET.root_dataset, 'pert')):
        create_dir(os.path.join(cfg.DATASET.root_dataset, 'pert'))

    main(cfg, args.gpu)
