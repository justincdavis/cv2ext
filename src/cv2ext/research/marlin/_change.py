# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cv2ext.bboxes._convert import yolo_to_xyxy

if TYPE_CHECKING:
    from typing_extensions import Self


class ChangeDetector:
    """ChangeDetector for Marlin methodology."""

    def __init__(self, path: str | Path) -> None:
        """
        Create a ChangeDetector instance.
        
        Parameters
        ----------
        path : str, Path
            The path to the weights of the forest.
    
        Raises
        ------
        ValueError
            If the loaded joblib file is not a RandomForestClassifier

        """
        self._forest: RandomForestClassifier = joblib.load(Path(path))
        if not isinstance(self._forest, RandomForestClassifier):
            err_msg = "ChangeDetector model must be RandomForestClassifier"
            raise ValueError(err_msg)

    @staticmethod
    def preprocess(
        image: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]] | list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        frame = image.copy()
        for data in detections:
            if len(data) == 3:
                (x1, y1, x2, y2), _, _ = data
            else:
                x1, y1, x2, y2 = data
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        resized_colored_image = cv2.resize(frame, (128, 128))
        hist_red: np.ndarray = cv2.calcHist(
            [resized_colored_image],
            [0],
            None,
            [256],
            [0, 256],
        )
        hist_green: np.ndarray = cv2.calcHist(
            [resized_colored_image],
            [1],
            None,
            [256],
            [0, 256],
        )
        hist_blue: np.ndarray = cv2.calcHist(
            [resized_colored_image],
            [2],
            None,
            [256],
            [0, 256],
        )

        feature_vector: np.ndarray = resized_colored_image.reshape(1, -1)
        feature_vector = feature_vector.astype(float)

        vectors: list[np.ndarray] = [
            feature_vector,
            hist_red.flatten(),
            hist_green.flatten(),
            hist_blue.flatten(),
        ]

        return np.concatenate(vectors, axis=None)  # type: ignore[assignment]

    @staticmethod
    def train(
        data_dir: str | Path,
        output_path: str | Path,
        n_estimators: int = 50,
        max_depth: int = 20,
    ) -> RandomForestClassifier:
        """
        Train the ChangeDetector on images and labels.
        
        Parameters
        ----------
        data_dir : str, Path
            The directory containing the images and labels in YOLO format.
            The directory should contain an images and a labels subdirectory.
            Example:
            data_dir
              | -> image
              | -> labels
        output_path : str, Path
            The output path to save the classifier.
        n_estimators : int
            The number of estimators to put inside the random forest.
            By default, 50
        max_depth : int
            The maximum depth of the random forest.
            By default, 20

        Returns
        -------
        RandomForestClassifier
            The underlying forest classifier

        Raises
        ------
        FileNotFoundError
            If the images and labels subdirectories can not be found
        FileNotFoundError
            If there are no matching images and labels

        """
        data_dir = Path(data_dir)
        image_dir = data_dir / "images"
        label_dir = data_dir / "labels"

        # ensure they exist
        if not image_dir.exists() or not label_dir.exists():
            err_msg = "Ensure both images and labels subdirectories are present."
            raise FileNotFoundError(err_msg)

        image_exts = [".jpg", ".jpeg", ".png"]
        image_names = [n.stem for n in image_dir.iterdir() if n.suffix in image_exts]
        label_names = [n.stem for n in label_dir.iterdir() if n.suffix in [".txt"]]
        common_names: list[str] = list(set(image_names).intersection(set(label_names)))

        if len(common_names) < 1:
            err_msg = "Could not find any matching images and labels."
            raise FileNotFoundError(err_msg)

        # helper function    
        def _load_vectors(name: str) -> list[tuple[np.ndarray, bool]] | None:
            try:
                label_path: Path = label_dir / f"{name}.txt"
                image_path: Path | None = None
                for ext in image_exts:
                    potential = image_dir / f"{name}{ext}"
                    if potential.exists():
                        image_path = potential
                        break
                if image_path is None:
                    return None

                image = cv2.imread(str(image_path))
                height, width = image.shape[:2]
                with label_path.open("r") as f:
                    labels = f.readlines()
                raw_labels = [map(float, label.split(" ")) for label in labels]
                yolo_labels = [(x, y, w, h) for _, x, y, w, h in raw_labels]
                bboxes = [yolo_to_xyxy(bbox, width, height) for bbox in yolo_labels]

                resized_img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
                vectors = ChangeDetector.preprocess(resized_img, bboxes)
                # unpack vectors for training
                unpacked = [
                    (vectors[0], False), 
                    (vectors[1], True),
                    (vectors[2], True),
                    (vectors[3], True),
                ]
                return unpacked
            except IndexError:
                return None

        # for each image/label pair fit the RandomForest
        forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        for cname in common_names:
            vectors = _load_vectors(cname)
            if vectors is None:
                continue

            features, labels = zip(*vectors)
            forest.fit(features, labels)

        # save the forest to a file
        output_path = Path(output_path)
        joblib.dump(forest, output_path)

        return forest

    def __call__(
        self: Self,
        image: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]] | list[tuple[int, int, int, int]],
    ) -> bool:
        return self.run(image, detections)

    def run(
        self: Self,
        image: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]] | list[tuple[int, int, int, int]],
    ) -> bool:
        return self._forest.predict([ChangeDetector.preprocess(image, detections)])[0]
