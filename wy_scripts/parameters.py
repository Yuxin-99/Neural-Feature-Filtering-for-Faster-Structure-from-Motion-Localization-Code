import os
import re


class Parameters(object):

    def __init__(self, dataset_path, session_id, method):
        # session_id = int(re.search(r'\d+', dataset_path).group())
        self.session_id = session_id
        self.method = method

        base_path = os.path.join(dataset_path, "exmaps_data")
        self.train_database_path = os.path.join(base_path, "base/database.db")
        self.train_points3D_path = os.path.join(base_path, "base/model/points3D.bin")
        self.train_model_cameras_path = os.path.join(base_path, "base/model/cameras.bin")

        self.query_db_path = os.path.join(base_path, "gt/database.db")
        self.query_img_folder = os.path.join(base_path, "gt/images/session_" + self.session_id)

        saved_data_path = os.path.join(base_path, method + "_saved_data")
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)
        self.results_path = os.path.join(base_path, method + "_results/")
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.avg_descs_base_path = os.path.join(saved_data_path, "avg_descs_base.npy")
        self.matches_save_path = os.path.join(saved_data_path, "matches.npy")
        self.match_times_save_path = os.path.join(self.results_path, "matching_times.npy")
        self.ratio_test_val = 0.85

        self.poses_save_path = os.path.join(saved_data_path, "poses.npy")
        est_img_path = os.path.join(self.results_path, "est_pnts_on_img")
        if not os.path.exists(est_img_path):
            os.makedirs(est_img_path)
        self.est_img_save_path = os.path.join(self.results_path, "est_pnts_on_img/")
        self.pose_rot_err_save_path = os.path.join(self.results_path, "rot_err.npy")
        self.pose_translation_err_save_path = os.path.join(self.results_path, "trans_err.npy")

        # not the session ones!
        self.query_gt_img_bin_path = os.path.join(base_path, "gt/model/images.bin")
        self.query_points_bin_path = os.path.join(base_path, "gt/model/points3D.bin")
        self.query_images_path = os.path.join(base_path, "gt/query_name.txt")

        # -------------------- ML classification --------------------
        self.live_db_path = os.path.join(base_path, "live/database.db")
        self.live_model_images_path = os.path.join(base_path, "live/model/images.bin")
        self.live_model_points3D_path = os.path.join(base_path, "live/model/points3D.bin")

        # make sure you delete the databases (.db) file first! and "ML_data" folder has to be created manually
        ml_db_dir = os.path.join(dataset_path, "ML_data/")
        # ml_db_dir = os.path.join(dataset_path, "ML_xy_data/")
        os.makedirs(ml_db_dir, exist_ok=True)
        self.ml_db_path = os.path.join(ml_db_dir, "ml_database_all.db")
        self.rf_model_path = os.path.join(ml_db_dir, "random_forest.joblib")
        self.rf_ml_metrics_path = os.path.join(ml_db_dir, "rf_metrics.txt")
        self.rf_xy_model_path = os.path.join(ml_db_dir, "random_forest_xy.joblib")
        self.rf_xy_ml_metrics_path = os.path.join(ml_db_dir, "ml_metrics_xy.txt")

        svm_dir = os.path.join(ml_db_dir, "svm/")
        os.makedirs(svm_dir, exist_ok=True)
        self.svm_model_path = os.path.join(svm_dir, "svm.joblib")
        self.svm_ml_metrics_path = os.path.join(svm_dir, "svm_metrics.txt")
        self.X_memmap_path = os.path.join(svm_dir, "X_memmap.npy")
        self.Y_memmap_path = os.path.join(svm_dir, "Y_memmap.npy")

        self.sgd_model_path = os.path.join(ml_db_dir, "sgd.joblib")
        self.sgd_ml_metrics_path = os.path.join(ml_db_dir, "sgd_metrics.txt")

        self.report_path = os.path.join(base_path, method + "_result_report.txt")

