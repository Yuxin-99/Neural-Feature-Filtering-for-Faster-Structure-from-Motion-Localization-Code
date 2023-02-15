# Refer to COLMAP/database.py and
# Neural-Feature-Filtering-for-Faster-Structure-from-Motion-Localization-Code/database.py

import numpy as np
import sqlite3
from sqlite3 import Error
import sys

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2                 # TODO: why times the max_image_id


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_file_path):
        return sqlite3.connect(database_file_path, factory=COLMAPDatabase)

    @staticmethod
    def array_to_blob(array):
        if IS_PYTHON3:
            return array.tostring()
        else:
            return np.getbuffer(array)

    @staticmethod
    def blob_to_array(blob, dtype, shape=(-1,)):
        if IS_PYTHON3:
            return np.fromstring(blob, dtype=dtype).reshape(*shape)
        else:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])                 # TODO: why keypoint shape??

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (self.array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (self.array_to_blob(descriptors),))

    def add_matches(self, img_id1, img_id2, matches):
        assert (len(matches.shape) == 2)
        assert (matches.shape[1] == 2)

        if img_id1 > img_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(img_id1, img_id2)
        matches = np.array(matches, np.uint32)

        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id, ) + matches.shape + (self.array_to_blob(matches),))

    @staticmethod
    def create_db_for_all_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                    image_id INTEGER NOT NULL,
                                                    name TEXT NOT NULL,
                                                    sift BLOB NOT NULL,
                                                    xyz BLOB NOT NULL,
                                                    xy BLOB NOT NULL,
                                                    rgb BLOB NOT NULL,
                                                    matched INTEGER NOT NULL
                                                );"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute(sql_drop_table_if_exists)
            conn.commit()
            conn.execute(sql_create_data_table)
            conn.commit()
            return conn
        except Error as e:
            print(e)

