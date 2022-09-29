# This file will aim to create the data for the RF model from Match no Match, MnM - (2020) paper
# then data.py, which is the file that fetches the data for training models
# It will overwrite existing data (db).
# To create data for all datasets:
# python3 create_training_data_match_no_match.py colmap_data/CMU_data/slice3 & python3 create_training_data_match_no_match.py colmap_data/CMU_data/slice4 & python3 create_training_data_match_no_match.py colmap_data/CMU_data/slice6 & python3 create_training_data_match_no_match.py colmap_data/CMU_data/slice10 & python3 create_training_data_match_no_match.py colmap_data/CMU_data/slice11 & python3 create_training_data_match_no_match.py colmap_data/Coop_data/slice1/ &

import os
import sys

import cv2

from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
from random import sample, choice
from tqdm import tqdm
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary

def get_image_decs(db, image_id): #not to be confused with get_queryDescriptors() in feature_matching_generator.py - that one normalises descriptors.
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs

def compute_kp_scales(kp_data):
    scales = np.empty([kp_data.shape[0],1])
    for i in range(kp_data.shape[0]):
        a11 = kp_data[i][2]
        a12 = kp_data[i][3]
        a21 = kp_data[i][4]
        a22 = kp_data[i][5]
        scale = (np.sqrt(a11 * a11 + a21 * a21) + np.sqrt(a12 * a12 + a22 * a22)) / 2
        scales[i,:] = scale
    return scales

def compute_kp_orientations(kp_data):
    orientations = np.empty([kp_data.shape[0],1])
    for i in range(kp_data.shape[0]):
        a11 = kp_data[i][2]
        # a12 = kp_data[i][3]
        a21 = kp_data[i][4]
        # a22 = kp_data[i][5]
        orientation = np.arctan2(a21, a11)
        orientations[i,:] = orientation
    return orientations

def get_image_keypoints_data(db, img_id):
    kp_data = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
    cols = kp_data[1]
    rows = kp_data[0]
    kp_data = COLMAPDatabase.blob_to_array(kp_data[2], np.float32)
    kp_data = kp_data.reshape([rows, cols])
    # Note: https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
    # https://github.com/colmap/colmap/issues/1219
    # https://stackoverflow.com/questions/45159314/decompose-2d-transformation-matrix
    kp_scales = compute_kp_scales(kp_data)
    kp_orientations = compute_kp_orientations(kp_data)
    xy = kp_data[:,0:2]
    return np.c_[xy, kp_scales, kp_orientations]

def createDataForMatchNoMatchMatchabilityComparison(image_base_dir, image_live_dir, db, live_images, output_db_path):
    print("Creating data..")
    training_data_db = COLMAPDatabase.create_db_match_no_match_data(os.path.join(output_db_path, "training_data.db"))
    training_data_db.execute("BEGIN")
    for img_id, img_data in tqdm(live_images.items()): #localised only , not the db ones
        if(('session_6' in img_data.name) == True):
            live_image_path = os.path.join(image_live_dir, img_data.name)
        else:
            live_image_path = os.path.join(image_base_dir, img_data.name) #then it is a base model image (still call it live for convinience)

        live_image_file = cv2.imread(live_image_path)
        keypoints_data = get_image_keypoints_data(db, img_id)
        descs = get_image_decs(db, img_id)

        assert len(img_data.point3D_ids) == keypoints_data.shape[0]
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0] == keypoints_data.shape[0]) # just for my sanity

        for i in range(img_data.xys.shape[0]): # can loop through descs or img_data.xys - same thing
            xy = img_data.xys[i] #np.float64, same as xyz
            current_point3D_id = img_data.point3D_ids[i]
            live_image_file_xy_green_intensity = live_image_file[int(xy[1]), int(xy[0])][1] #reverse indexing

            if(current_point3D_id == -1): # means feature is unmatched
                matched = 0
            else:
                matched = 1

            desc = descs[i] # np.uint8
            desc_scale = keypoints_data[i, 2]
            desc_orientation = keypoints_data[i, 3]

            training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (img_id,) + (COLMAPDatabase.array_to_blob(desc),) + (matched,) + (desc_scale,) +
                          (desc_orientation,) + (xy[0],) + (xy[1],) + (int(live_image_file_xy_green_intensity),))

    print("Committing..")
    training_data_db.commit()
    print("Generating Stats..")
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total descs: " + str(len(stats)))
    print("Total matched descs: " + str(len(matched)))
    print("Total unmatched descs: " + str(len(unmatched)))
    print("% of matched decs: " + str(len(matched) * 100 / len(unmatched)))

    print("Done!")

base_path = sys.argv[1]
print("Base path: " + base_path)
parameters = Parameters(base_path)
db_live = COLMAPDatabase.connect(parameters.live_db_path)
live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)
image_live_dir = os.path.join(base_path, 'live/images/')
image_base_dir = os.path.join(base_path, 'base/images/')

db_live_path = os.path.join(base_path, "live/database.db")
output_path = os.path.join(base_path, "match_or_no_match_comparison_data")
os.makedirs(output_path, exist_ok = True)
createDataForMatchNoMatchMatchabilityComparison(image_base_dir, image_live_dir, db_live, live_model_images, output_path)
