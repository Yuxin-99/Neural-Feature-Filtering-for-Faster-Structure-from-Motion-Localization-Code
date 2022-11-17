import os


class Parameters(object):

    def __init__(self, base_path, dataset_id):
        self.train_database_path = os.path.join(base_path, "database" + dataset_id + ".db")
        self.train_points3D_path = os.path.join(base_path, "sparse/points3D.bin")
        self.train_model_cameras_path = os.path.join(base_path, "sparse/cameras.bin")

        self.query_db_path = os.path.join(base_path, "query" + dataset_id + ".db")
        self.query_img_folder = os.path.join(base_path, "query")

        self.avg_descs_base_path = os.path.join(base_path, "avg_descs_base.npy")
        self.matches_save_path = os.path.join(base_path, "saved_data/matches.npy")
        self.match_times_save_path = os.path.join(base_path, "saved_data/matching_times.npy")

        self.poses_save_path = os.path.join(base_path, "saved_data/poses.npy")
        self.est_img_save_path = os.path.join(base_path, "est_pnts_on_img/")
        self.pose_rot_err_save_path = os.path.join(base_path, "results/rot_err.npy")
        self.pose_translation_err_save_path = os.path.join(base_path, "results/trans_err.npy")

        # not the session ones!
        self.query_gt_img_bin_path = os.path.join(base_path, "sparse_query/images.bin")
        self.query_camera_poses_folder = os.path.join(base_path, "camera-poses")
        self.query_images_path = os.path.join(base_path, "gt/query_name.txt")

        self.ratio_test_val = 0.85

