"""
This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""


from copy import deepcopy
from typing import List, Optional

import numpy as np
from default_settings import (
    BoostTrackPlusPlusSettings,
    BoostTrackSettings,
    GeneralSettings,
)
from tracker.assoc import (
    MhDist_similarity,
    associate,
    iou_batch,
    shape_similarity,
    soft_biou_batch,
)
from tracker.ecc import ECC


class BoostTrack:
    def __init__(self, video_name: Optional[str] = None):
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []

        self.max_age = GeneralSettings.max_age(video_name)
        self.iou_threshold = GeneralSettings["iou_threshold"]
        self.det_thresh = GeneralSettings["det_thresh"]
        self.min_hits = GeneralSettings["min_hits"]

        self.lambda_iou = BoostTrackSettings["lambda_iou"]
        self.lambda_mhd = BoostTrackSettings["lambda_mhd"]
        self.lambda_shape = BoostTrackSettings["lambda_shape"]
        self.use_dlo_boost = BoostTrackSettings["use_dlo_boost"]
        self.use_duo_boost = BoostTrackSettings["use_duo_boost"]
        self.dlo_boost_coef = BoostTrackSettings["dlo_boost_coef"]

        self.use_rich_s = BoostTrackPlusPlusSettings["use_rich_s"]
        self.use_sb = BoostTrackPlusPlusSettings["use_sb"]
        self.use_vt = BoostTrackPlusPlusSettings["use_vt"]

        if GeneralSettings["use_embedding"]:
            self.embedder = EmbeddingComputer(
                GeneralSettings["dataset"], GeneralSettings["test_dataset"], True,
            )
        else:
            self.embedder = None

        if GeneralSettings["use_ecc"]:
            self.ecc = ECC(scale=350, video_name=video_name, use_cache=True)
        else:
            self.ecc = None

    def update(self, dets, img_tensor, img_numpy, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if dets is None:
            return np.empty((0, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        self.frame_count += 1

        # Rescale
        scale = min(
            img_tensor.shape[2] / img_numpy.shape[0],
            img_tensor.shape[3] / img_numpy.shape[1],
        )
        dets = deepcopy(dets)
        dets[:, :4] /= scale

        if self.ecc is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.camera_update(transform)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        confs = np.zeros((len(self.trackers), 1))

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(
                dets, self.use_rich_s, self.use_sb, self.use_vt,
            )

        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)

        remain_inds = dets[:, 4] >= self.det_thresh
        dets = dets[remain_inds]
        scores = dets[:, 4]

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        emb_cost = None
        if self.embedder and dets.size > 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)
            trk_embs = []
            for t in range(len(self.trackers)):
                trk_embs.append(self.trackers[t].get_emb())
            trk_embs = np.array(trk_embs)
            if trk_embs.size > 0 and dets.size > 0:
                emb_cost = (
                    dets_embs.reshape(dets_embs.shape[0], -1)
                    @ trk_embs.reshape((trk_embs.shape[0], -1)).T
                )
        emb_cost = None if self.embedder is None else emb_cost

        matched, unmatched_dets, unmatched_trks, sym_matrix = associate(
            dets,
            trks,
            self.iou_threshold,
            mahalanobis_distance=self.get_mh_dist_matrix(dets),
            track_confidence=confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
        )

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = 0.95
        dets_alpha = af + (1 - af) * (1 - trust)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], scores[m[0]])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                self.trackers.append(KalmanBoxTracker(dets[i, :], emb=dets_embs[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate((d, [trk.id + 1], [trk.get_confidence()])).reshape(
                        1, -1,
                    ),
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(
        self, detections: np.ndarray, buffered: bool = False,
    ) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return (
            iou_batch(detections, trackers)
            if not buffered
            else soft_biou_batch(detections, trackers)
        )

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1,))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(
                np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims]),
            )

        return (
            (z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2
            * sigma_inv.reshape((1, -1, n_dims))
        ).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(
                    len(boost_detections),
                )
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(
                        np.intersect1d(
                            boost_detections_args[args], boost_detections_args[tmp],
                        ),
                        boost_detections_args[boxi],
                    )

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(
                            remaining_boxes.tolist() + [boost_detections_args[boxi]],
                        )

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(
        self,
        detections: np.ndarray,
        use_rich_sim: bool,
        use_soft_boost: bool,
        use_varying_th: bool,
    ) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [
                pos[0],
                pos[1],
                pos[2],
                pos[3],
                0,
                self.trackers[t].time_since_update - 1,
            ]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(
                    detections[:, 4],
                    alpha * detections[:, 4] + (1 - alpha) * max_s ** (1.5),
                )
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (
                    np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e) < S
                ).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections
