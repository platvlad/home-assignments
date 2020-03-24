#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import (CornerStorage, FrameCorners)
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    TriangulationParameters,
    build_correspondences,
    pose_to_view_mat3x4,
    calc_inlier_indices,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    check_baseline,
    get_baseline,
    _remove_correspondences_with_ids)


class TrackingError(Exception):
    pass


class CameraTracker:
    def __init__(self, intrinsic_mat, corner_storage, known_view_1, known_view_2):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        frame_num_1 = known_view_1[0]
        frame_num_2 = known_view_2[0]
        self.pc_builder = PointCloudBuilder()
        view_matr_1 = pose_to_view_mat3x4(known_view_1[1])
        view_matr_2 = pose_to_view_mat3x4(known_view_2[1])

        self.tracked_views = [eye3x4()] * len(self.corner_storage)
        self.tracked_views[frame_num_1] = view_matr_1
        self.tracked_views[frame_num_2] = view_matr_2
        self.initial_baseline = get_baseline(view_matr_1, view_matr_2)
        print("initial baseline =", self.initial_baseline)

        self._add_points_from_frame(frame_num_1, frame_num_2, initial_triangulation=True)
        self.min_init_frame = min(frame_num_1, frame_num_2)
        self.max_init_frame = max(frame_num_1, frame_num_2)
        self.outliers = set()

    def _get_pos(self, frame_number) -> np.ndarray:
        corners = self.corner_storage[frame_number]
        _, pts_ids, corners_ids = np.intersect1d(self.pc_builder.ids, corners.ids, assume_unique=True,
                                                 return_indices=True)

        common_pts_3d = self.pc_builder.points[pts_ids]
        common_ids = self.pc_builder.ids[pts_ids]
        common_corners = corners.points[corners_ids]
        inlier_mask = np.zeros_like(common_pts_3d, dtype=np.bool)
        inlier_counter = 0
        for common_id, mask_elem in zip(common_ids[:, 0], inlier_mask):
            if common_id not in self.outliers:
                mask_elem[:] = True
                inlier_counter += 1
        if inlier_counter > 7:
            common_pts_3d = common_pts_3d[inlier_mask].reshape(-1, 3)
            common_ids = common_ids[inlier_mask[:, :1]].reshape(-1, 1)
            common_corners = common_corners[inlier_mask[:, :2]].reshape(-1, 2)
        max_reproj_error = 3.0
        if len(common_pts_3d) < 4:
            raise TrackingError('Not enough points to solve RANSAC on frame ' + str(frame_number))
        _, rot_vec, tr_vec, inliers = cv2.solvePnPRansac(common_pts_3d,
                                                         common_corners,
                                                         self.intrinsic_mat,
                                                         None,
                                                         reprojectionError=max_reproj_error,
                                                         flags=cv2.SOLVEPNP_EPNP)
        extrinsic_mat = rodrigues_and_translation_to_view_mat3x4(rot_vec, tr_vec)
        proj_matr = self.intrinsic_mat @ extrinsic_mat
        if inliers is None:
            raise TrackingError('Failed to solve PnP on frame' + str(frame_number))

        while len(inliers) < 8 and max_reproj_error < 50:
            max_reproj_error *= 1.2
            inliers = calc_inlier_indices(common_pts_3d, common_corners, proj_matr, max_reproj_error)
        inlier_pts = common_pts_3d[inliers]
        inlier_corners = common_corners[inliers]
        outlier_ids = np.setdiff1d(common_ids, common_ids[inliers], assume_unique=True)
        self.outliers.update(outlier_ids)
        if len(inliers) < 4:
            inlier_pts = common_pts_3d
            inlier_corners = common_corners
        print('Found position on', len(inlier_corners), 'inliers')
        _, rot_vec, tr_vec, inliers = cv2.solvePnPRansac(inlier_pts, inlier_corners, self.intrinsic_mat, None,
                                                         rot_vec, tr_vec, useExtrinsicGuess=True)
        return rodrigues_and_translation_to_view_mat3x4(rot_vec, tr_vec)

    def _add_points_from_frame(self,
                               frame_num_1: int,
                               frame_num_2: int,
                               initial_triangulation: bool = False) -> int:
        corners_1 = self.corner_storage[frame_num_1]
        corners_2 = self.corner_storage[frame_num_2]
        correspondences = build_correspondences(corners_1, corners_2, ids_to_remove=self.pc_builder.ids)
        if len(correspondences.ids) > 0:
            max_reproj_error = 1.0
            min_angle = 1.0
            view_1 = self.tracked_views[frame_num_1]
            view_2 = self.tracked_views[frame_num_2]
            triangulation_params = TriangulationParameters(max_reproj_error, min_angle, 0)
            pts_3d, triangulated_ids, med_cos = triangulate_correspondences(correspondences, view_1,
                                                                            view_2, self.intrinsic_mat,
                                                                            triangulation_params)
            if initial_triangulation:
                num_iter = 0
                while len(pts_3d) < 8 and len(correspondences.ids) > 7 and num_iter < 100:
                    max_reproj_error *= 1.2
                    min_angle *= 0.8
                    triangulation_params = TriangulationParameters(max_reproj_error, min_angle, 0)
                    pts_3d, triangulated_ids, med_cos = triangulate_correspondences(correspondences, view_1,
                                                                                    view_2, self.intrinsic_mat,
                                                                                    triangulation_params)
                    num_iter += 1

                if num_iter >= 100 and len(pts_3d) < 4:
                    raise TrackingError('Failed to triangulate enough points')
            self.pc_builder.add_points(triangulated_ids, pts_3d)

            return len(pts_3d)
        elif initial_triangulation:
            raise TrackingError('Not found correspondences on image pair')
        else:
            return 0

    def track(self):
        curr_frame = self.min_init_frame + 1
        seq_size = len(self.corner_storage)
        for _ in range(2, seq_size):
            if curr_frame == self.max_init_frame:
                curr_frame += 1
            if curr_frame >= seq_size:
                curr_frame = self.min_init_frame - 1
                self.outliers = set()
            print('Frame', curr_frame)
            try:
                self.tracked_views[curr_frame] = self._get_pos(curr_frame)
            except TrackingError as error:
                print(error)
                print('Stopping tracking')
                break
            prev_curr_diff = 5
            num_pairs = 0
            num_added = 0
            while num_pairs < 5:
                prev_frame = \
                    curr_frame - prev_curr_diff if curr_frame > self.min_init_frame else curr_frame + prev_curr_diff
                prev_curr_diff += 1
                if prev_frame < seq_size and (self.min_init_frame <= prev_frame or curr_frame < self.min_init_frame):
                    if check_baseline(self.tracked_views[prev_frame], self.tracked_views[curr_frame],
                                      self.initial_baseline * 0.15):
                        num_added += self._add_points_from_frame(prev_frame, curr_frame)
                        num_pairs += 1
                else:
                    break
            print('Added', num_added, 'points')
            print('Point cloud size = ', len(self.pc_builder.ids))
            if curr_frame > self.min_init_frame:
                curr_frame += 1
            else:
                curr_frame -= 1

        return self.tracked_views, self.pc_builder


def _test_frame_pair(intrinsic_mat: np.ndarray,
                     corners_1: FrameCorners,
                     corners_2: FrameCorners,
                     param_koeff: float = 1) -> Tuple[int, Optional[Pose]]:
    correspondences = build_correspondences(corners_1, corners_2)
    if len(correspondences.ids) < 6:
        return 0, None
    points2d_1 = correspondences.points_1
    points2d_2 = correspondences.points_2

    essential, essential_inliers = cv2.findEssentialMat(points2d_1, points2d_2, intrinsic_mat, threshold=param_koeff)
    homography, homography_inliers = cv2.findHomography(points2d_1, points2d_2, method=cv2.RANSAC)
    if len(np.where(homography_inliers > 0)[0]) > len(np.where(essential_inliers > 0)[0]):
        return 0, None
    if essential.shape != (3, 3):
        return 0, None
    num_passed, rot, t, mask = cv2.recoverPose(essential, points2d_1, points2d_2,
                                               intrinsic_mat, mask=essential_inliers)

    outlier_ids = np.array(
        [pt_id for pt_id, mask_elem in zip(correspondences.ids, mask) if mask_elem[0] == 0],
        dtype=correspondences.ids.dtype)
    inlier_correspondences = _remove_correspondences_with_ids(correspondences, outlier_ids)
    if len(inlier_correspondences.ids) < 4:
        return 0, None
    view_matr_1 = eye3x4()
    view_matr_2 = np.hstack((rot, t))
    triangulation_params = TriangulationParameters(param_koeff, 1.0 / param_koeff, 0.0)
    pts_3d, trianulated_ids, med_cos = triangulate_correspondences(inlier_correspondences, view_matr_1, view_matr_2,
                                                                   intrinsic_mat, triangulation_params)
    if len(pts_3d) < 4:
        return 0, None
    print(len(inlier_correspondences.ids), len(pts_3d))
    return len(pts_3d), view_mat3x4_to_pose(view_matr_2)


def _find_pos_pair(intrinsic_mat: np.ndarray,
                   corner_storage: CornerStorage,
                   param_koeff: float = 1) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    best_num_pts = 0
    best_frame = 1
    best_pose = None
    for frame_number in range(1, len(corner_storage)):
        print(frame_number)
        num_pts, pose = _test_frame_pair(intrinsic_mat, corner_storage[0], corner_storage[frame_number])
        if num_pts > best_num_pts:
            best_num_pts = num_pts
            best_pose = pose
            best_frame = frame_number
    if best_pose is None:
        if param_koeff < 4:
            return _find_pos_pair(intrinsic_mat, corner_storage, param_koeff * 2)
        raise TrackingError("Failed to initialize tracking")
    return (0, view_mat3x4_to_pose(eye3x4())), (best_frame, best_pose)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = [eye3x4()] * len(corner_storage), PointCloudBuilder()

    try:
        known_view_1, known_view_2 = _find_pos_pair(intrinsic_mat, corner_storage)
        tracker = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2)
        view_mats, point_cloud_builder = tracker.track()
    except TrackingError as error:
        print(error)
        print('Poses and point cloud are not calculated')

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
