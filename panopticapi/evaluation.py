#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import time
import pickle
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id

OFFSET = 256 * 256 * 256
VOID = 0

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]

def gt_ids_to_classes(gt):
    gt_copy = np.array(gt)
    gt_copy[gt > 1000] = (gt[gt > 1000] // 1000)
    return gt_copy

def pred_ids_to_classes(pred):
    pred_copy = np.array(pred)
    pred_copy[pred > 1000] = (pred[pred > 1000] // 1000)

    pred_copy[pred_copy == 18] = 33
    pred_copy[pred_copy == 17] = 31
    pred_copy[pred_copy == 15] = 28
    pred_copy[pred_copy == 14] = 27
    pred_copy[pred_copy == 13] = 26
    pred_copy[pred_copy == 12] = 25
    pred_copy[pred_copy == 11] = 24
    pred_copy[pred_copy == 10] = 0
    pred_copy[pred_copy == 9] = 22
    pred_copy[pred_copy == 8] = 21
    pred_copy[pred_copy == 7] = 20
    pred_copy[pred_copy == 5] = 17
    pred_copy[pred_copy == 4] = 13
    pred_copy[pred_copy == 2] = 11
    pred_copy[pred_copy == 1] = 7

    return pred_copy


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.uece = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.uece += pq_stat_cat.uece
        return self

num_bins = 10

class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)
        self.spa_uece = 0
        self.sem_uece = 0
        self.uece_curve = np.zeros(num_bins)
        self.num_frames = 0
        self.segments = 0

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n, pece = 0, 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            uece = self.pq_per_cat[label].uece
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            pece_class = uece / tp if tp != 0 else 0
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'pECE': pece_class,
                                        'spa_uECE': 0, 'sem_uECE': 0}
            pq += pq_class
            sq += sq_class
            rq += rq_class
            pece += pece_class

        stats_dict = {'pq': pq / n, 'sq': sq / n, 'rq': rq / n,
                      'pECE': pece / n,
                      'spa_uECE': self.spa_uece / self.num_frames,
                      'sem_uECE': self.sem_uece / self.num_frames,
                      'uECE_curve': self.uece_curve / self.segments,
                      'n': n}

        return stats_dict, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    bounds = np.linspace(0, 1, num_bins + 1)

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        with open(os.path.join(pred_folder, pred_ann['unc_name']), 'rb') as unc_file:
            unc_img = pickle.load(unc_file)

        with open(os.path.join(pred_folder, pred_ann['spa_unc_name']), 'rb') as spa_unc_file:
            spa_unc_img = pickle.load(spa_unc_file)

        with open(os.path.join(pred_folder, pred_ann['sem_unc_name']), 'rb') as sem_unc_file:
            sem_unc_img = pickle.load(sem_unc_file)

        spa_conf = 1 - spa_unc_img
        sem_conf = 1 - sem_unc_img

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        gt_classes = gt_ids_to_classes(pan_gt)
        pred_classes = pred_ids_to_classes(pan_pred)

        sem_correctness = gt_classes == pred_classes

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        spa_correctness = np.zeros_like(sem_correctness)

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            segment_idxs = (pan_gt_pred == (gt_label * OFFSET + pred_label))
            sem_correct_pixels = sem_correctness[segment_idxs]
            unc_pixels = unc_img[segment_idxs]
            conf_pixels = 1 - unc_pixels

            spa_correctness[segment_idxs] = 1

            num_in_seg = np.sum(segment_idxs)

            seg_uece = 0

            bin_occupants = np.zeros(num_bins)

            for i in range(num_bins):
                lower = bounds[i]
                upper = bounds[i + 1]
                bin_idxs = (conf_pixels >= lower) & (conf_pixels < upper)
                num_in_bin = np.sum(bin_idxs)
                mean_correct = np.mean(sem_correct_pixels[bin_idxs])
                mean_conf = np.mean(conf_pixels[bin_idxs])
                abs_diff = np.abs(mean_correct - mean_conf)
                seg_uece += np.nan_to_num(((num_in_bin / num_in_seg) * abs_diff))

                bin_occupants[i] = np.nan_to_num(mean_correct)

            pq_stat.uece_curve += bin_occupants
            pq_stat.segments += 1

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                pq_stat[gt_segms[gt_label]['category_id']].uece += seg_uece
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        pixels_cared_about = gt_classes != VOID
        spa_num_in_img = np.sum(pixels_cared_about)
        sem_num_in_img = np.sum(pixels_cared_about)

        for i in range(num_bins):
            lower = bounds[i]
            upper = bounds[i + 1]

            spa_bin_idxs = (spa_conf >= lower) & (spa_conf < upper)
            spa_num_in_bin = np.sum(spa_bin_idxs)
            spa_mean_correct = np.mean(spa_correctness[pixels_cared_about & spa_bin_idxs])
            spa_mean_conf = np.mean(spa_conf[pixels_cared_about & spa_bin_idxs])
            spa_abs_diff = np.abs(spa_mean_correct - spa_mean_conf)
            pq_stat.spa_uece += np.nan_to_num(((spa_num_in_bin / spa_num_in_img) * spa_abs_diff))

        for i in range(num_bins):
            lower = bounds[i]
            upper = bounds[i + 1]

            sem_bin_idxs = (sem_conf >= lower) & (sem_conf < upper)
            sem_num_in_bin = np.sum(sem_bin_idxs)
            sem_mean_correct = np.mean(sem_correctness[pixels_cared_about & sem_bin_idxs])
            sem_mean_conf = np.mean(sem_conf[pixels_cared_about & sem_bin_idxs])
            sem_abs_diff = np.abs(sem_mean_correct - sem_mean_conf)
            pq_stat.sem_uece += np.nan_to_num(((sem_num_in_bin / sem_num_in_img) * sem_abs_diff))

        pq_stat.num_frames += 1

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    # pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)
    pq_stat = pq_compute_single_core(1, matched_annotations_list, gt_folder, pred_folder, categories)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>8s}  {:>8s} {:>5s}".format("", "PQ", "SQ", "RQ", "pECE", "spa_uECE", "sem_uECE", "N"))
    print("-" * (10 + 7 * 7))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f}  {:5.1f}    {:5.1f}    {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            100 * results[name]['pECE'],
            100 * results[name]['spa_uECE'],
            100 * results[name]['sem_uECE'],
            results[name]['n'])
        )

    print(results['All']['uECE_curve'])

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    args = parser.parse_args()
    pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)
