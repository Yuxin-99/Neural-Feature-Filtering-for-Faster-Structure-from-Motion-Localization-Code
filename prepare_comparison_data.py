import tensorflow as tf
from tensorflow import keras
from database import COLMAPDatabase
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark

print("Setting up...")
# the "gt" here means "after_epoch_data" pretty much
db_gt_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/after_epoch_database.db"
query_images_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/new_model/images.bin"
query_images_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/images_list.txt"
query_cameras_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/new_model/cameras.bin"
live_points3D_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/original_live_data/model/points3D.bin"

db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do not exist in the live db!
query_images = read_images_binary(query_images_bin_path)
query_images_names = load_images_from_text_file(query_images_path)
localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)
query_images_ground_truth_poses = get_query_images_pose_from_images(localised_query_images_names, query_images)

# live points
# Note: you will need to run this first, "get_points_3D_mean_desc_single_model.py" - to get the 3D points avg descs from the LIVE model.
# use the folder original_live_data/ otherwise you will be using the epoch image/3D point descriptors if you use the new_model
# also you will need the scale between the colmap poses and the ARCore poses (for 2020-06-22 the 392 images are from morning run)
# Matching will happen from the query images (epoch images) on the live model, otherwise if you use the epoch model it will be "cheating"
# as the descriptors from the epoch images that you are trying to match will already be in the epoch model. Just use the epoch model for ground truth pose errors comparisons.
train_descriptors_live = np.load('colmap_data/Coop_data/slice1/ML_data/after_epoch_data/original_live_data/avg_descs.npy').astype(np.float32)
points3D_live = read_points3d_default(live_points3D_bin_path) #note the order of the gt 3d points are different - because of COLMAP..
points3D_xyz_live = get_points3D_xyz(points3D_live)

K = get_intrinsics_from_camera_bin(query_cameras_bin_path, 3)  # 3 because 1 -base, 2 -live, 3 -epoch images
# for ar_core data
ar_core_poses_path = 'colmap_data/Coop_data/slice1/ML_data/after_epoch_data/test_images/2020-06-22/arcore_poses/'
colmap_poses_path = query_images_bin_path  # just for clarity purposes
scale = calc_scale_COLMAP_ARCORE(ar_core_poses_path, colmap_poses_path)
print("Scale: " + str(scale))

print("Feature matching random and vanillia descs..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 0.9  # as previous publication
# random 80 ones - why 80 ?
random_no = 80  # Given these features are random the errors later on will be much higher, and benchmarking might fail because there will be < 4 matches sometimes
random_matches = feature_matcher_wrapper(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose=True, random_limit=random_no)
# all of them as in first publication (should be around 800 for each image)
vanillia_matches = feature_matcher_wrapper(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose=True)

# get the benchmark data here for random features and the 800 from previous publication - will return the average values for each image
benchmarks_iters = 5

inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark(benchmarks_iters, ransac, random_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Time: %2.2f" % (inlers_no, outliers, iterations, time))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
random_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall])

inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark(benchmarks_iters, ransac, vanillia_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Time: %2.2f" % (inlers_no, outliers, iterations, time))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
vanillia_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall])

np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/query_images_ground_truth_poses.npy", query_images_ground_truth_poses)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/localised_query_images_names.npy", localised_query_images_names)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/points3D_xyz_live.npy", points3D_xyz_live)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/K.npy", K)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/scale.npy", scale)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/random_matches.npy", random_matches)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/vanillia_matches.npy", vanillia_matches)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/random_matches_data.npy", random_matches_data)
np.save("colmap_data/Coop_data/slice1/ML_data/comparison_data/vanillia_matches_data.npy", vanillia_matches_data)