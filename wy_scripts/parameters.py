import os


class Parameters(object):

    def __init__(self, base_path, dataset_id):
        self.train_database_path = os.path.join(base_path, "database" + dataset_id + ".db")
        self.train_points3D_path = os.path.join(base_path, "sparse/points3D.bin")

        self.query_db_path = os.path.join(base_path, "query" + dataset_id + ".db")
        self.query_img_folder = os.path.join(base_path, "query")

        self.matches_save_path = os.path.join(base_path, "matches.npy")

        self.poses_save_path = os.path.join(base_path, "poses.npy")

        self.avg_descs_base_path = os.path.join(base_path, "avg_descs_base.npy")

        # not the session ones!
        self.query_images_path = os.path.join(base_path , "gt/query_name.txt")

        self.gt_model_cameras_path = os.path.join(base_path , "gt/model/cameras.bin")

        self.ratio_test_val = 0.85

