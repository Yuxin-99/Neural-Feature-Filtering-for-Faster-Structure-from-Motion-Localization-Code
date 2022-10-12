import numpy as np
import os
import sys

from database import COLMAPDatabase
from read_model import read_points3D_binary, get_points3D_xyz
from feature_matching import feature_matcher_wrapper

# python3 main.py ../CMU-models/sparse/points3D.bin ../CMU-models/query2.db ../CMU-models/query_images ../CMU-models/avg_descs_base.npy


def main():
    # read the arguments of model file paths
    # path to the points3D.bin from the training model
    points3D_file_path = sys.argv[1]
    # path to the database of the query images
    database_path = sys.argv[2]
    # path to the folder that contains the query images
    query_images_path = sys.argv[3]

    # assume the average descriptors are pre-computed by "get_points_3D_mean_desc.py" and saved to the file
    avg_descs_base_path = sys.argv[4]

    # db_gt = COLMAPDatabase.connect(database_path)                   # database of the training images
    # read the points3D out
    points3D = read_points3D_binary(points3D_file_path)
    points3D_xyz = get_points3D_xyz(points3D)

    # load the average descriptors from the npy file
    train_descriptors_base = np.load(avg_descs_base_path).astype(np.float32)

    db_query = COLMAPDatabase.connect(database_path)                # database of the query images
    query_images_names = os.listdir(query_images_path)
    # do the matching of training descriptors and the query images
    print("start to do matching!")
    matches = feature_matcher_wrapper(db_query, query_images_names, train_descriptors_base, points3D_xyz, 0.9, verbose=True)
    print("matching is done!")


if __name__ == "__main__":
    main()