import numpy as np
import os
import sys

from database import COLMAPDatabase
from feature_matching import feature_matcher_wrapper
from get_points_3D_mean_desc import compute_avg_desc
from kerasNN_model import get_kerasNN_model
from rf_model import get_rf_model
from svm_model import get_svm_model, get_sgd_model
from parameters import Parameters
from pose_estimator import do_pose_estimation
from pose_evaluator import evaluate_est_pose
from read_model import read_points3D_binary, get_points3D_xyz


# python3 main.py ../Dataset/slice7 0


def main():
    """ 1. Do feature matching between the query images and the point cloud provided by the training images of
        the dataset we are using.
        2. Use the matches to do pose estimation for each query image
        3. Compare the estimated poses and ground truth data to analyze the error metrics(rotation and translation)"""

    # construct the parameters
    base_path = sys.argv[1]  # e.g.: ../../Dataset/slice7
    session_id = sys.argv[2]  # session number in gt folder, e.g., 7
    method = sys.argv[3]  # e.g.: base, rf...
    params = Parameters(base_path, session_id, method)

    # Read the related files
    points3D_file_path = params.live_model_points3D_path  # points3D.bin from the training model (provided by dataset)
    query_database_path = params.query_db_path
    db_query = COLMAPDatabase.connect(query_database_path)  # database of the query images
    query_images_path = params.query_img_folder + "session_" + params.session_id + "/"  # path to the folder that contains the query images
    query_images_names = sorted(os.listdir(query_images_path))

    # load the saved matches data file to decide or do the feature matching first
    matching_timer_on = sys.argv[4] == "timer"  # if we need to record the time of feature matching
    filter_perc = "not available"
    if (os.path.exists(params.matches_save_path)) and (not matching_timer_on):
        matches = np.load(params.matches_save_path, allow_pickle=True).item()
        matching_time = np.load(params.match_times_save_path, allow_pickle=True).item()
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
            train_descriptors_base = compute_avg_desc(params.live_db_path, points3D_file_path)
            np.save(params.avg_descs_base_path, train_descriptors_base)

        # do the matching of training descriptors and the query images
        print("start to do matching!")
        do_filter, clf_model = get_filter_model(method, params)
        matches, matching_time, filter_percentage = feature_matcher_wrapper(db_query, query_images_names,
                                                                            train_descriptors_base,
                                                                            points3D_xyz, params, clf_model,
                                                                            verbose=True)  # lower thresh
        filter_perc = str(filter_percentage * 100) + "%"
        print("matching is done!")
        np.save(params.matches_save_path, matches)
        np.save(params.match_times_save_path, matching_time)

    # for i in range(len(query_images_names)):
    #     matrix = get_camera_matrix(db_query, query_images_names[i])

    # load the saved poses data file or do pose estimation
    if os.path.exists(params.poses_save_path):
        rt_poses = np.load(params.poses_save_path, allow_pickle=True).item()
        degenerate_pose_perc = (len(query_images_names) - len(rt_poses)) / len(query_images_names)
        print("Load poses from file " + params.poses_save_path + "!")
    else:
        print("start to do pose estimating")
        rt_poses, degenerate_pose_perc = do_pose_estimation(matches, query_images_names, query_images_path,
                                                            params.est_img_save_path)
        np.save(params.poses_save_path, rt_poses)
        print("Pose estimating is done!")

    # compute the error metrics for the estimated poses
    t_error, r_error, maa = evaluate_est_pose(rt_poses, params)
    pose_errs = [t_error, r_error, maa]
    avg_match_time = sum(matching_time.values()) / len(matching_time)
    record_result(params.report_path, params.slice_id, method, pose_errs, avg_match_time, degenerate_pose_perc, filter_perc)


# return bool: indicate if we need to filter non-matchable descriptors;
#        clf_model: the classifier model we will use for filtering
def get_filter_model(method, params):
    # assume the ml database is already created
    if method == "rf_rgb":
        return True, get_rf_model(params, 1, 1)
    elif method == "rf_xy":
        return True, get_rf_model(params, True, 0)
    elif method == "svm":
        return True, get_svm_model(params)
    elif method == "sgd":
        return True, get_sgd_model(params)
    elif method == "kerasNN":
        return True, get_kerasNN_model(params, 0)
    elif method == "kerasNN_rgb":
        return True, get_kerasNN_model(params, 1)
    else:
        return False, None


def record_result(report_path, slice_id, method, pose_err, match_time, degenerate_perc, filter_perc):
    with open(report_path, 'w') as f:
        f.write("Data slice: " + slice_id)
        f.write('\n')
        f.write("Method: " + method)
        f.write('\n')
        f.write("Average translation error: " + str(pose_err[0]) + " meters")
        f.write('\n')
        f.write("Average rotation error: " + str(pose_err[1]) + " degrees")
        f.write('\n')
        f.write("Mean Average Accuracy: " + str(pose_err[2]))
        f.write('\n')
        f.write("Average matching time: " + str(match_time) + " seconds")
        f.write('\n')
        f.write("Degenerate pose percentage: " + str(degenerate_perc * 100) + "%")
        f.write('\n')
        f.write("Filtered descriptor percentage: " + filter_perc)
        f.write('\n')


if __name__ == "__main__":
    main()
