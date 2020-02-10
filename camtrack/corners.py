#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _dist_to_corners(corners, s_corner):
    dists = np.linalg.norm(corners[:, 0, :] - s_corner, axis=1)
    return np.min(dists)


def _filter_new_corners(old_corners, old_ids, new_corners, new_ids, threshold):
    new_corners_space = np.zeros(shape=new_corners.shape)
    new_ids_space = np.zeros(shape=new_ids.shape)
    counter = len(old_ids)
    old_corners = np.concatenate((old_corners, new_corners_space))
    old_ids = np.concatenate((old_ids, new_ids_space))
    for new_corner, new_id in zip(new_corners, new_ids):
        min_dist = _dist_to_corners(old_corners, new_corner[0])
        if min_dist >= threshold:
            old_ids[counter] = new_id
            old_corners[counter] = new_corner
            counter += 1
    return old_corners[:counter], old_ids[:counter]


def filter_tracked_corners(old_ids, new_corners, status, err):
    tracked_ids = np.zeros(shape=(len(new_corners),))
    tracked_cv_corners = np.zeros(shape=(len(new_corners), 1, 2))
    counter = 0
    for i in range(len(new_corners)):
        if status[i] > 0 and err[i] < 5:
            tracked_ids[counter] = old_ids[i]
            tracked_cv_corners[counter] = new_corners[i]
            counter += 1
    return tracked_cv_corners[:counter], tracked_ids[:counter]


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    image_0 = image_0 * 255
    image_0 = image_0.astype(np.uint8)
    cv_corners = cv2.goodFeaturesToTrack(image_0, 500, 0.01, 10, blockSize=11, useHarrisDetector=False)
    max_id = 0
    ids = np.arange(max_id, max_id + len(cv_corners))
    max_id = max(max_id, ids[-1])
    corners = FrameCorners(
        ids,
        cv_corners,
        np.ones(shape=(len(cv_corners),)) * 5
    )
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = image_1 * 255
        image_1 = image_1.astype(np.uint8)
        next_corners = np.zeros(shape=np.shape(cv_corners), dtype=cv_corners.dtype)
        _, status, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, cv_corners, next_corners, winSize=(15, 15))
        tracked_cv_corners, tracked_ids = filter_tracked_corners(ids, next_corners, status, err)
        max_id = max(max_id, ids[-1])
        new_cv_corners = cv2.goodFeaturesToTrack(image_1, 500, 0.01, 10, blockSize=11, useHarrisDetector=False)
        new_ids = np.arange(max_id + 1, max_id + len(new_cv_corners) + 1)
        cv_corners, ids = _filter_new_corners(tracked_cv_corners, tracked_ids, new_cv_corners, new_ids, 10)
        max_id = max(max_id, ids[-1])
        cv_corners = cv_corners.astype(np.float32)
        corners = FrameCorners(
            ids,
            cv_corners,
            np.ones(shape=(len(cv_corners),)) * 5
        )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
