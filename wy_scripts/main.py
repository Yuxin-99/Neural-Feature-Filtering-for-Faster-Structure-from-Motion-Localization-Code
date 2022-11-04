import numpy as np
import os
import re
import sys

from database import COLMAPDatabase
from feature_matching import feature_matcher_wrapper
from get_points_3D_mean_desc import compute_avg_desc
from parameters import Parameters
from pose_estimator import do_pose_estimation
from pose_evaluator import evaluate_est_pose
from read_model import read_points3D_binary, get_points3D_xyz


# python3 main.py ../CMU-models/sparse/points3D.bin ../CMU-models/query_models/undistorted_query.db ../CMU-models/undistorted_query_imgs 1
#  (Total matches: 664518, no of images 1824. Average matches per image: 364.31907894736844)

# python3 main.py ../Dataset/slice7 0


def main():
    """ 1. Do feature matching between the query images and the point cloud provided by the training images of
        the dataset we are using.
        2. Use the matches to do pose estimation for each query image
        3. Compare the estimated poses and ground truth data to analyze the error metrics(rotation and translation)"""

    # construct the parameters
    base_path = sys.argv[1]                     # e.g.: ../Dataset/slice7
    # assume the dataset number is included in the directory name
    dataset_id = int(re.search(r'\d+', base_path).group())
    dataset_id = str(dataset_id)
    params = Parameters(base_path, dataset_id)

    # Read the related files
    points3D_file_path = params.train_points3D_path         # points3D.bin from the training model (provided by dataset)
    query_database_path = params.query_db_path
    db_query = COLMAPDatabase.connect(query_database_path)  # database of the query images
    query_images_path = params.query_img_folder             # path to the folder that contains the query images
    query_images_names = sorted(os.listdir(query_images_path))

    # load the saved matches data file to decide or do the feature matching first
    if os.path.exists(params.matches_save_path):
        matches = np.load(params.matches_save_path, allow_pickle=True).item()
        print("Load matches from file " + params.matches_save_path + "!")
    else:
        """do feature matching to find and save the matches first"""
        # read the points3D out
        points3D = read_points3D_binary(points3D_file_path)
        points3D_xyz = get_points3D_xyz(points3D)

        # load the average descriptors from the npy file or compute it first if not existed
        if os.path.exists(params.avg_descs_base_path):
            train_descriptors_base = np.load(params.avg_descs_base_path).astype(np.float32)
        else:
            train_descriptors_base = compute_avg_desc(params.train_database_path, params.train_points3D_path)
            np.save(params.avg_descs_base_path, train_descriptors_base)

        # do the matching of training descriptors and the query images
        print("start to do matching!")
        matches = feature_matcher_wrapper(db_query, query_images_names, train_descriptors_base, points3D_xyz,
                                          params.ratio_test_val, verbose=True)  # lower thresh
        print("matching is done!")
        np.save(params.matches_save_path, matches)

    # load the saved poses data file or do pose estimation
    if os.path.exists(params.poses_save_path):
        rt_poses = np.load(params.poses_save_path, allow_pickle=True).item()
        print("Load poses from file " + params.poses_save_path + "!")
    else:
        print("start to do pose estimating")
        rt_poses = do_pose_estimation(matches, query_images_names, query_images_path)
        np.save(params.poses_save_path, rt_poses)
        print("Pose estimating is done!")

    # compute the error metrics for the estimated poses
    gt_from_model = sys.argv[2] == "1"
    evaluate_est_pose(rt_poses, gt_from_model, params)


if __name__ == "__main__":
    main()
