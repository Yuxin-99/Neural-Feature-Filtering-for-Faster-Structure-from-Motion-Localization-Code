import os
import re


class Parameters(object):

    def __init__(self, dataset_path):
        dataset_id = int(re.search(r'\d+', dataset_path).group())
        self.dataset_id = str(dataset_id)

        base_path = os.path.join(dataset_path, "exmaps_data")
        self.train_database_path = os.path.join(base_path, "base/database.db")
        self.train_points3D_path = os.path.join(base_path, "base/model/points3D.bin")
        self.train_model_cameras_path = os.path.join(base_path, "base/model/cameras.bin")

        self.query_db_path = os.path.join(base_path, "gt/database.db")
        self.query_img_folder = os.path.join(base_path, "gt/images/session_" + self.dataset_id)

        saved_data_path = os.path.join(base_path, "flann_saved_data")
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)
        self.results_path = os.path.join(base_path, "flann_results/")
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.avg_descs_base_path = os.path.join(saved_data_path, "avg_descs_base.npy")
        self.matches_save_path = os.path.join(saved_data_path, "matches.npy")
        self.match_times_save_path = os.path.join(self.results_path, "matching_times.npy")

        self.poses_save_path = os.path.join(saved_data_path, "poses.npy")
        est_img_path = os.path.join(self.results_path, "est_pnts_on_img")
        if not os.path.exists(est_img_path):
            os.makedirs(est_img_path)
        self.est_img_save_path = os.path.join(self.results_path, "est_pnts_on_img/")
        self.pose_rot_err_save_path = os.path.join(self.results_path, "rot_err.npy")
        self.pose_translation_err_save_path = os.path.join(self.results_path, "trans_err.npy")

        # not the session ones!
        self.query_gt_img_bin_path = os.path.join(base_path, "gt/model/images.bin")
        # self.query_camera_poses_folder = os.path.join(base_path, "camera-poses")
        # self.query_images_path = os.path.join(base_path, "gt/query_name.txt")

        self.live_db_path = os.path.join(base_path, "live/database.db")
        self.live_model_images_path = os.path.join(base_path, "live/model/images.bin")
        self.live_model_points3D_path = os.path.join(base_path, "live/model/points3D.bin")

        self.ratio_test_val = 0.85

