# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from cv2ext.bboxes._algorithms import match
from cv2ext.bboxes._iou import iou

from ._confgraph import build_graph

if TYPE_CHECKING:
    from collections.abc import Callable

_log = logging.getLogger(__name__)


def _characterize(
    model: Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]],
    modelname: str,
    power_draw: float,
    output_dir: Path,
    image_dir: Path,
    image_names: list[str],
    ground_truth: list[list[tuple[int, int, int, int]]],
    num_bins: int = 10,
    *,
    overwrite: bool | None = None,
) -> None:
    """
    Characterize the given model.

    This function characterizes the given model and saves the
    characterization to the given output directory.

    Parameters
    ----------
    model : Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]
        The model to characterize.
        Output assumed to be in the x1, y1, x2, y2 format.
    modelname : str
        The name of the model to characterize.
    output_dir : Path
        The directory to save the characterization to.
    power_draw : float
        The power draw for performing inference with the model.
        This should be measured ahead of time by chosen method.
    image_dir : Path
        The directory containing the images to use for characterization.
    image_names : list[str]
        The list of image names to use for characterization.
    ground_truth : list[tuple[tuple[int, int, int, int], int]]
        The ground truth bounding boxes and classes for the images.
        Bounding boxes are assumed to be in the x1, y1, x2, y2 format.
    num_bins : int, optional
        The number of bins to use for binning the data, by default 10
    overwrite : bool, optional
        Whether or not to overwrite an existing characterization if
        data files already exist. By default, True, will overwrite

    Raises
    ------
    FileNotFoundError
        If the image directory does not exist.
    RuntimeError
        If bins and indices cannot be computed.
        If confidence value is greater than bin key.
        If confidence value is less than bin key.

    """
    if overwrite is None:
        overwrite = True

    if not Path.exists(output_dir / modelname):
        Path.mkdir(output_dir / modelname, parents=True)

    csvpath = output_dir / modelname / f"{modelname}.csv"
    if csvpath.exists() and not overwrite:
        _log.debug(f"Skipping characterization for: {modelname}")

    jsonpath = output_dir / modelname / f"{modelname}.json"
    if not Path.exists(jsonpath):
        with Path.open(jsonpath, "w") as f:
            json.dump({"power_draw": power_draw}, f, indent=4)

    if not Path.exists(image_dir):
        err_msg = f"Image directory {image_dir} does not exist."
        raise FileNotFoundError(err_msg)

    image_stats: dict[str, dict[str, float]] = defaultdict(dict)

    _log.debug("Forming image stats")
    for image_name, gt in zip(image_names, ground_truth):
        imagepath = image_dir / image_name
        image = cv2.imread(str(imagepath))

        t0 = time.perf_counter()
        outputs = model(image)
        t1 = time.perf_counter()
        t_inference = t1 - t0

        # compute the mean iou and mean conf
        bboxes = []
        scores = []
        for bbox, score, _ in outputs:
            bboxes.append(bbox)
            scores.append(score)

        # match bounding boxes to ground truth labels
        matches = match(bboxes, gt, iou_threshold=0.05)

        # compute iou for each match idx
        if len(matches) == 0:
            if len(gt) == 0:
                mean_iou = 1.0
                mean_score = 1.0
            else:
                mean_iou = 0.0
                mean_score = 0.0
        else:
            ious = [iou(bboxes[i], gt[j]) for i, j in matches]
            mean_iou = float(np.mean(ious))
            mean_score = float(np.mean(scores))

        # compute the recall
        if len(gt) == 0:
            recall = float(len(matches) == len(gt))
        else:
            recall = len(matches) / len(gt)

        image_stats[image_name]["time"] = t_inference
        image_stats[image_name]["iou"] = mean_iou
        image_stats[image_name]["recall"] = recall
        image_stats[image_name]["conf"] = mean_score
        image_stats[image_name]["energy"] = t_inference * power_draw

    fieldnames = list(image_stats[image_names[0]].keys())

    _log.debug("Forming CSV")
    with Path.open(output_dir / modelname / f"{modelname}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", *fieldnames])
        writer.writeheader()
        for image_name, image_data in image_stats.items():
            entry = {"filename": image_name, **image_data}
            writer.writerow(entry)

    with Path.open(jsonpath, "r") as f:
        json_data = json.load(f)

    # raw metrics
    json_data["power_draw"] = str(power_draw)
    for metric in fieldnames:
        raw_data = [data[metric] for data in image_stats.values()]
        json_data[metric] = {
            "mean": str(np.mean(raw_data)),
            "median": str(np.median(raw_data)),
            "var": str(np.var(raw_data)),
            "std": str(np.std(raw_data)),
            "min": str(np.min(raw_data)),
            "max": str(np.max(raw_data)),
        }

    # correlations
    conf_data = np.array([data["conf"] for data in image_stats.values()])
    iou_data = np.array([data["iou"] for data in image_stats.values()])
    recall_data = np.array([data["recall"] for data in image_stats.values()])
    json_data["conf_corr"] = str(np.mean(np.corrcoef(conf_data, iou_data)))
    json_data["conf_corr_recall"] = str(np.mean(np.corrcoef(conf_data, recall_data)))

    # binning
    json_bin_data: dict[str, int | float | dict] = {}
    json_bin_data["num_bins"] = num_bins
    bin_array = np.array([b / num_bins for b in range(num_bins + 1)])
    indices = np.digitize(conf_data, bin_array)

    if len(indices) != len(conf_data):
        err_msg = "Could not compute bins and indices."
        raise RuntimeError(err_msg)

    binned_iou = defaultdict(list)
    binned_conf = defaultdict(list)
    binned_recall = defaultdict(list)

    for idx, indice in enumerate(indices):
        conf_key = indice / num_bins
        iou_val = iou_data[idx]
        conf_val = conf_data[idx]
        recall_val = recall_data[idx]
        # # verify confidence checks
        # if conf_val > conf_key:
        #     err_msg = "Confidence value is greater than bin key."
        #     raise RuntimeError(err_msg)
        # if conf_val < conf_key - 1 / num_bins:
        #     err_msg = "Confidence value is less than bin key."
        #     raise RuntimeError(err_msg)
        binned_iou[conf_key].append(iou_val)
        binned_conf[conf_key].append(conf_val)
        binned_recall[conf_key].append(recall_val)

    possible_bins = [b / num_bins for b in range(num_bins + 1)]
    for conf_bin in possible_bins:
        iou_bin_data = binned_iou[conf_bin]
        conf_bin_data = binned_conf[conf_bin]
        recall_bin_data = binned_recall[conf_bin]
        if len(iou_bin_data) == 0:
            iou_bin_data = [0.0, 0.0]
            conf_bin_data = [0.0, 0.0]
            recall_bin_data = [0.0, 0.0]
        alpha = np.mean(
            [i / c for i, c in zip(iou_bin_data, conf_bin_data) if c != 0],
        )
        alpha_recall = np.mean(
            [r / c for r, c in zip(recall_bin_data, conf_bin_data) if c != 0],
        )
        if str(alpha) == "nan":
            alpha = 1.0
        if str(alpha_recall) == "nan":
            alpha_recall = 1.0

        z = np.array([1.0, 0.0])
        if not (np.sum(conf_bin_data) == 0 or np.sum(iou_bin_data) == 0):
            z = np.polyfit(conf_bin_data, iou_bin_data, 1)
        conf_bin_corr = float(np.mean(np.corrcoef(conf_bin_data, iou_bin_data)))
        z_recall = np.array([1.0, 0.0])
        if not (np.sum(conf_bin_data) == 0 or np.sum(recall_bin_data) == 0):
            z_recall = np.polyfit(conf_bin_data, recall_bin_data, 1)
        conf_bin_corr_recall = float(
            np.mean(np.corrcoef(conf_bin_data, recall_bin_data)),
        )

        if str(conf_bin_corr) == "nan":
            conf_bin_corr = 1.0
        if str(conf_bin_corr_recall) == "nan":
            conf_bin_corr_recall = 1.0

        sub_bin_data = {}
        sub_bin_data["alpha"] = str(alpha)
        sub_bin_data["conf_corr"] = str(conf_bin_corr)
        sub_bin_data["iou_mean"] = str(np.mean(iou_bin_data))
        sub_bin_data["conf_mean"] = str(np.mean(conf_bin_data))
        sub_bin_data["fit"] = str(z.tolist())
        sub_bin_data["alpha_recall"] = str(alpha_recall)
        sub_bin_data["conf_corr_recall"] = str(conf_bin_corr_recall)
        sub_bin_data["recall_mean"] = str(np.mean(recall_bin_data))
        sub_bin_data["fit_recall"] = str(z_recall.tolist())
        # add to the json
        json_bin_data[str(conf_bin)] = sub_bin_data
    json_data["bins"] = json_bin_data

    # write final json file
    with Path.open(jsonpath, "w") as f:
        json.dump(json_data, f, indent=4)


def characterize(
    models: list[
        Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]
    ],
    model_names: list[str],
    power_draws: list[float],
    output_dir: Path,
    image_dir: Path,
    image_files: list[str],
    ground_truth: list[list[tuple[int, int, int, int]]],
    num_bins: int = 10,
    min_connects: int = 1,
    cutoff: float = 0.0,
    upper_outlier_cutoff: float = 80.0,
    connectivity: int = 2,
    graph_type: type[nx.Graph | nx.DiGraph] = nx.Graph,
    *,
    purge_connectivity: bool | None = None,
    overwrite: bool | None = None,
) -> None:
    """
    Characterize the given models.

    This function characterizes the given models and saves the
    characterization to the given output directory.

    Parameters
    ----------
    models : list[Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]]
        The list of models to characterize.
        Each model represented by a single callable.
    model_names : list[str]
        The list of model names to use for the models.
    power_draws : list[float]
        The power draw for performing inference with the model.
        This should be measured ahead of time by chosen method.
    output_dir : Path
        The directory to save the characterization to.
    image_dir : Path
        The directory containing the images to use for characterization.
    image_files : list[str]
        The list of image names to use for characterization.
    ground_truth : list[list[tuple[tuple[int, int, int, int], int]]]
        The ground truth bounding boxes and classes for the images.
        Bounding boxes are assumed to be in the x1, y1, x2, y2 format.
    num_bins : int, optional
        The number of bins to use for binning the data, by default 10
    min_connects : int, optional
        The minimum number of connections to use for the graph, by default 1
    cutoff : float, optional
        The cutoff to use for the graph, by default 0.0
    upper_outlier_cutoff : float, optional
        The upper outlier cutoff to use for the graph, by default 80.0
    connectivity : int, optional
        The connectivity to use for the graph, by default 2
    graph_type : type[nx.Graph | nx.DiGraph], optional
        The type of graph to use, by default None
        If None, will use nx.Graph.
    purge_connectivity : bool, optional
        If True, the cutoff will be iteratively increased until the graph
        has the minimum number of connected components.
        By default None, which will not purge the graph.
    overwrite : bool, optional
        Whether or not to overwrite an existing characterization if
        data files already exist. By default, True, will overwrite

    """
    if not Path.exists(output_dir):
        Path.mkdir(output_dir, parents=True)

    for model, model_name, power_draw in zip(models, model_names, power_draws):
        _log.debug(f"Characterizing model: {model_name}")

        # run the characterization
        _characterize(
            model=model,
            modelname=model_name,
            output_dir=output_dir,
            power_draw=power_draw,
            image_dir=image_dir,
            image_names=image_files,
            ground_truth=ground_truth,
            num_bins=num_bins,
            overwrite=overwrite,
        )

    _log.debug("Building conf graph...")
    build_graph(
        output_dir=output_dir,
        num_bins=num_bins,
        min_connects=min_connects,
        cutoff=cutoff,
        upper_outlier_cutoff=upper_outlier_cutoff,
        connectivity=connectivity,
        graph_type=graph_type,
        purge_connectivity=purge_connectivity,
    )
