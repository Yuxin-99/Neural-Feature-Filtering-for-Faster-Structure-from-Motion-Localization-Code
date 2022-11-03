# The difference form "get_points_3D_mean_desc_single_model.py" or "get_points_3D_mean_descs.py" as it returns the SIFT descs and coordinate x,y,z points.
# And also I do not normalise the descs here.

# one liner (for weatherwax server, ogg etc, or the docker in them)
# using relative paths here (CMU or Coop)
#  arguments
# 1 - live db
# 2 - live images
# 3 - live points3D
# 4 - output filename
# python3 get_points_3D_mean_desc.py ../CMU-models/train.db ../CMU-models/sparse/points3D.bin ../CMU-models/undistorted_avg_descs_base.npy
# python3 get_points_3D_mean_desc.py ../slice7/database7.db ../slice7/sparse/points3D.bin ../slice7/avg_descs_base.npy

import sys
import numpy as np
from database import COLMAPDatabase
from read_model import read_points3D_binary


def compute_avg_desc(db_path, points3D_path):
    """ path to the database of the training model and points3D.bin of the training model"""
    db = COLMAPDatabase.connect(db_path)
    points3D = read_points3D_binary(points3D_path)

    no = 0
    desc_mean = np.empty([0, 128])
    # points_info = np.empty([0, 131])                        # (SIFT + xyz)

    for i, v in points3D.items():
        no += 1
        point_id = v.id
        points3D_descs = np.empty([0, 128])
        points_image_ids = points3D[point_id].image_ids     # COLMAP adds the image twice some times.
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each img_id.
        for k in range(len(points_image_ids)):
            print("Point: " + str(no) + ", Image: " + str(k), end="\r")
            img_id = points_image_ids[k]
            data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(img_id) + "'")
            data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
            descs_rows = int(np.shape(data)[0] / 128)
            descs = data.reshape([descs_rows, 128])         # descs for the whole image
            keypoint_index = points3D[point_id].point2D_idxs[k]
            # keypoints and descs are ordered the same (so I use the point2D_idxs to index descs)
            desc = descs[keypoint_index]
            # this is the desc of keypoint with index, keypoint_index, from image with id, img_id.
            desc = desc.reshape(1, 128)
            points3D_descs = np.r_[points3D_descs, desc]

        # adding and calculating the mean here!
        mean = points3D_descs.mean(axis=0).reshape(1, 128)
        # row = np.append(mean, v.xyz)
        # points_info = np.r_[points_info, row.reshape(1, 131)]
        desc_mean = np.r_[desc_mean, mean]

    print()
    return desc_mean





