# This file will aim to create the data for the RF model from Match no Match, MnM - (2020) paper
# It will extract OpenCV features and insert them in colmaps database and run the triangulator again
# I clear the old data, and keep the poses.
# You will need to tun this on the CYENS machine as it has pycolmap and colmap installed
# To start we need to copy base,live,gt to CYENS then run this script for each base,live,gt ; (scp -r -P 11568 base live gt  alex@4.tcp.eu.ngrok.io:/home/alex/uni/models_for_match_no_match/CMU_slice_3/)

import glob
import os
import sys
import cv2
import pycolmap
import shutil
import colmap
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
import random
from tqdm import tqdm
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary, get_valid_images_ids_from_db, get_image_name_from_db_with_id, get_all_images_names_from_db


def empty_points_3D_txt_file(path):
    open(path, 'w').close()

def arrange_images_txt_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if "#" in line or ".jpg" in line:
                f.write(line)
            else:
                f.write("\n")

def countDominantOrientations(keypoints):
    domOrientations = np.ones([len(keypoints),1])
    for i in range(len(keypoints)):
        if(domOrientations[i,0] == 1): # if point has not been checked before
            dominantsIndex = np.zeros([len(keypoints),1])
            dominantsIndex[i,0] = 1
            nDominants = 1
        for j in range(i+1, len(keypoints), 1):
            dist = np.abs(keypoints[i].pt[0] - keypoints[j].pt[0]) + np.abs(keypoints[i].pt[1] - keypoints[j].pt[1])
            if(dist == 0.0):
                nDominants +=1
                dominantsIndex[j, 0] = 1
        for k in range(len(dominantsIndex)):
            if(dominantsIndex[k,0] == 1):
                domOrientations[k,0] = nDominants
    return domOrientations

base_path = sys.argv[1] #/home/alex/uni/models_for_match_no_match/CMU_slice_3/
model_base_path = os.path.join(base_path, "base")
model_live_path = os.path.join(base_path, "live")
model_gt_path = os.path.join(base_path, "gt")
images_base_path = os.path.join(model_base_path, "images")
images_live_path = os.path.join(model_live_path, "images")
images_gt_path = os.path.join(model_gt_path, "images")
sift = cv2.SIFT_create()

# New TODO:
# look at cmu_sparse_reconstuctor.py, for help

# Note: use images names from database to locate them for opencv feature extraction

# 1 - replace base model features with openCV sift
manually_created_model_txt_path = os.path.join(model_base_path,'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
os.makedirs(manually_created_model_txt_path, exist_ok=True)
# set up files as stated online in COLMAP's faq
colmap_model_path = os.path.join(model_base_path, 'model/0')
reconstruction = pycolmap.Reconstruction(colmap_model_path)
# export model to txt
reconstruction.write_text(manually_created_model_txt_path)
points_3D_file_txt_path = os.path.join(manually_created_model_txt_path, 'points3D.txt')
images_file_txt_path = os.path.join(manually_created_model_txt_path, 'images.txt')
empty_points_3D_txt_file(points_3D_file_txt_path)
arrange_images_txt_file(images_file_txt_path)
base_db_path = os.path.join(model_base_path, 'database.db')

base_db = COLMAPDatabase.connect(base_db_path)
image_names = get_all_images_names_from_db(base_db)

if(base_db.dominant_orientations_column_exists() == False):
    base_db.add_dominant_orientations_column()
    base_db.commit() #we need to commit here

print("Extracting data from images..")
for image_name in image_names:
    image_file_path = os.path.join(images_base_path, image_name)
    img = cv2.imread(image_file_path)
    kps_plain = []
    kps, des = sift.detectAndCompute(img,None)
    dominant_orientations = countDominantOrientations(kps)

    kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
    kps_plain = np.array(kps_plain)
    base_db.replace_keypoints(image_name, kps_plain, dominant_orientations)
    base_db.replace_descriptors(image_name, des)
    base_db.commit()

base_db.delete_all_matches()
base_db.delete_all_two_view_geometries()
base_db.commit()

colmap.vocab_tree_matcher(base_db)

# 2 - triangulate the base model -> base opencv sift model
opencv_sift_base_model_path = os.path.join(model_base_path, 'output_opencv_sift_model')
colmap.point_triangulator(base_db_path, images_base_path, manually_created_model_txt_path, opencv_sift_base_model_path)

print('Base Model done!')

# 3 - replace the live database's features with opencv sift features (for all images, base + live)
# 4 - register the new live images against the base opencv sift model

# 5 - replace the gt database's features with opencv sift features (for all images, base + live + gt)
# 6 - register the new gt images against the live opencv sift model

# Note: 09/10/2022 Code inefficient below

# copy all images in one folder for feature extraction
# os.makedirs(all_images_path, exist_ok = True)
# files = glob.iglob(os.path.join(images_base_path, "*.jpg"))
# for file in files:
#     if(os.path.exists(file) == False):
#         shutil.copy(file, all_images_path)
# files = glob.iglob(images_live_path + "/**/*.jpg")
# for file in files:
#     if(os.path.exists(file) == False):
#         shutil.copy(file, all_images_path)
# files = glob.iglob(images_gt_path + "/**/*.jpg")
# for file in files:
#     if (os.path.exists(file) == False):
#         shutil.copy(file, all_images_path)

# models = [ "live", "gt"] #modify this if you want to run for base, live, or gt
# sift = cv2.SIFT_create()
#
# for model in tqdm(models):
#     model_path = os.path.join(base_path, model)
#     model_images_path = os.path.join(model_path, "images")
#     match_no_match_db_path = os.path.join(model_path, 'match_no_match_database.db')
#     db_path = os.path.join(model_path, 'database.db')
#     base_db_path = os.path.join(base_path, 'base/database.db')
#     output_model_path = os.path.join(model_path, 'output_opencv_sift_model')
#     os.makedirs(output_model_path, exist_ok=True)
#
#     if(model == 'live' or model == 'gt'): #only register images
#         query_image_names = open(os.path.join(model_path, 'query_name.txt'), 'r').readlines() #this is to make sure the image name from the db is from live or gt images only, use the original .txt file
#         query_image_names = [query_image_name.strip() for query_image_name in query_image_names]
#         colmap_model_path = os.path.join(model_path, 'model')
#     else: #base, triangulate opecv sift model
#         manually_created_model_txt_path = os.path.join(model_path,'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
#         os.makedirs(manually_created_model_txt_path, exist_ok=True)
#         #  set up files as stated online in COLMAP's faq
#         points_3D_file_txt_path = os.path.join(manually_created_model_txt_path, 'points3D.txt')
#         images_file_txt_path = os.path.join(manually_created_model_txt_path, 'images.txt')
#         empty_points_3D_txt_file(points_3D_file_txt_path)
#         arrange_images_txt_file(images_file_txt_path)
#         colmap_model_path = os.path.join(model_path, 'model/0')
#         reconstruction = pycolmap.Reconstruction(colmap_model_path)
#         # export model to txt
#         reconstruction.write_text(manually_created_model_txt_path)
#         query_image_names = None
#
#     # TODO: 09/10/2022, This is a mess. run triangulation for base, then break the loop and write seperate methods for live and gt
#     # also make sure you use the new base opencv sift model to register live images
#     #  recopy the data files from server here
#
#     shutil.copy(base_db_path, match_no_match_db_path)
#     db_match_no_match = COLMAPDatabase.connect(match_no_match_db_path)
#
#     # image_ids = get_valid_images_ids_from_db(db_match_no_match, query_image_names)
#     # if query_image_names != None:
#     #     print(f'query_image_names no. is: {len(query_image_names)}')
#     #     assert len(image_ids) == len(query_image_names)
#
#     if(db_match_no_match.dominant_orientations_column_exists() == False):
#         db_match_no_match.add_dominant_orientations_column()
#         db_match_no_match.commit() #we need to commit here
#
#     print("Extracting data from images..")
#     image_names = ['img_01314_c0_1288106344867943us.jpg', 'img_01426_c0_1288106355199722us.jpg', 'img_01397_c0_1288106352400127us.jpg']
#     for i in range(227,229,1):
#         print(i)
#         image_name = image_names[i-227] #or db here
#         image_file_path = os.path.join(all_images_path, image_name)
#         img = cv2.imread(image_file_path)
#         kps_plain = []
#         kps, des = sift.detectAndCompute(img,None)
#         dominant_orientations = countDominantOrientations(kps)
#
#         kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
#         kps_plain = np.array(kps_plain)
#         db_match_no_match.replace_keypoints(i, kps_plain, dominant_orientations)
#         db_match_no_match.replace_descriptors(i, des)
#         db_match_no_match.commit()
#         breakpoint()
#
#     db_match_no_match.delete_all_matches()
#     db_match_no_match.delete_all_two_view_geometries()
#     db_match_no_match.commit()
#
#     print("Matching..")
#     new_query_image_names_file_path = os.path.join(model_path, f'query_name_{model}.txt') #new will contain absolute paths
#     if(model == 'live' or model == 'gt'):
#         with open(new_query_image_names_file_path, 'w') as f:
#             for filename in glob.glob(model_images_path + '/**/*'):
#                 f.write(f"{filename}\n")
#         breakpoint()
#         colmap.vocab_tree_matcher(match_no_match_db_path, new_query_image_names_file_path)
#     else: #base
#         colmap.vocab_tree_matcher(match_no_match_db_path)
#
#     if (model == 'live' or model == 'gt'):
#         print("Registering images on base model..")
#         base_model_path = os.path.join(base_path, 'base/output_opencv_sift_model')
#         # registering images on the base model, using the opencv sift features and saving the opencv sift live model in the output_model_path
#         breakpoint()
#         colmap.image_registrator(match_no_match_db_path, base_model_path, output_model_path)
#     else: #base, here we create to model to start with (with opencv sift)
#         colmap.point_triangulator(match_no_match_db_path, model_images_path, manually_created_model_txt_path, output_model_path)
#
#     db_match_no_match.close()
# print("Done!")

# old code
#
# def get_image_decs(db, image_id): #not to be confused with get_queryDescriptors() in feature_matching_generator.py - that one normalises descriptors.
#     data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
#     data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
#     descs_rows = int(np.shape(data)[0] / 128)
#     descs = data.reshape([descs_rows, 128])  # descs for the whole image
#     return descs
#
# def compute_kp_scales(kp_data):
#     scales = np.empty([kp_data.shape[0],1])
#     for i in range(kp_data.shape[0]):
#         a11 = kp_data[i][2]
#         a12 = kp_data[i][3]
#         a21 = kp_data[i][4]
#         a22 = kp_data[i][5]
#         scale = (np.sqrt(a11 * a11 + a21 * a21) + np.sqrt(a12 * a12 + a22 * a22)) / 2
#         scales[i,:] = scale
#     return scales
#
# def compute_kp_orientations(kp_data):
#     orientations = np.empty([kp_data.shape[0],1])
#     for i in range(kp_data.shape[0]):
#         a11 = kp_data[i][2]
#         # a12 = kp_data[i][3]
#         a21 = kp_data[i][4]
#         # a22 = kp_data[i][5]
#         orientation = np.arctan2(a21, a11)
#         orientations[i,:] = orientation
#     return orientations
#
# def get_image_keypoints_data(db, img_id):
#     kp_data = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
#     cols = kp_data[1]
#     rows = kp_data[0]
#     kp_data = COLMAPDatabase.blob_to_array(kp_data[2], np.float32)
#     kp_data = kp_data.reshape([rows, cols])
#     # Note: https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
#     # https://github.com/colmap/colmap/issues/1219
#     # https://stackoverflow.com/questions/45159314/decompose-2d-transformation-matrix
#     kp_scales = compute_kp_scales(kp_data)
#     kp_orientations = compute_kp_orientations(kp_data)
#     xy = kp_data[:,0:2]
#     return np.c_[xy, kp_scales, kp_orientations]
#
# def get_full_path(lpath, bpath, name):
#     if (('session' in name) == True):
#         image_path = os.path.join(lpath, name)
#     else:
#         image_path = os.path.join(bpath, name)  # then it is a base model image (still call it live for convinience)
#     return image_path
#
# def get_subset_of_pairs(all_pair_ids, no):
#     random.shuffle(all_pair_ids)
#     img_left_ids = []
#     img_right_ids = []
#     pair_ids = []
#     pbar = tqdm(total=no)
#     while (len(pair_ids) < no):
#         rnd_pair_id = random.choice(all_pair_ids)
#         img_id_1, img_id_2 = pair_id_to_image_ids(rnd_pair_id[0])
#         if ((img_id_1 not in img_left_ids) and (img_id_2 not in img_right_ids)):
#             pair_ids.append(rnd_pair_id)  # which is a tuple
#             pbar.update(1)
#     pbar.close()
#     return pair_ids
#
# def createDataForMatchNoMatchMatchabilityComparison(image_base_dir, image_live_dir, db_live, live_images, output_db_path, pairs_limit=-1):
#     print("Getting Pairs")
#     if(pairs_limit == -1):
#         pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
#     else:
#         all_pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
#         pair_ids = get_subset_of_pairs(all_pair_ids, pairs_limit) #as in paper
#
#     print("Creating data..")
#     training_data_db = COLMAPDatabase.create_db_match_no_match_data(os.path.join(output_db_path, "training_data_small.db"))
#     training_data_db.execute("BEGIN")
#
#     for pair in tqdm(pair_ids):
#         pair_id = pair[0]
#         img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)
#
#         # when a db image has not been localised ...
#         if((img_id_1 not in live_images) or (img_id_2 not in live_images)):
#             continue
#
#         img_1_file_name = live_images[img_id_1].name
#         img_2_file_name = live_images[img_id_2].name
#
#         img_1_file = cv2.imread(get_full_path(image_live_dir, image_base_dir, img_1_file_name))
#         img_2_file = cv2.imread(get_full_path(image_live_dir, image_base_dir, img_2_file_name))
#
#         descs_img1 = get_image_decs(db_live, img_id_1)
#         descs_img2 = get_image_decs(db_live, img_id_2)
#
#         pair_data = db_live.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
#         rows = pair_data[0]
#
#         if(rows < 1): #no matches in this pair, no idea why COLMAP stores it...
#             continue
#
#         cols = 2 #for each image
#         zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
#         zero_based_indices_left = zero_based_indices[:, 0]
#         zero_based_indices_right = zero_based_indices[:, 1]
#
#         keypoints_data_img_1 = get_image_keypoints_data(db_live, img_id_1)
#         keypoints_data_img_2 = get_image_keypoints_data(db_live, img_id_2)
#
#         keypoints_data_img_1_matched = keypoints_data_img_1[zero_based_indices_left]
#         keypoints_data_img_1_matched_descs = descs_img1[zero_based_indices_left]
#         keypoints_data_img_1_unmatched = np.delete(keypoints_data_img_1, zero_based_indices_left, axis=0)
#         keypoints_data_img_1_unmatched_descs = np.delete(descs_img1, zero_based_indices_left, axis=0)
#
#         keypoints_data_img_2_matched = keypoints_data_img_2[zero_based_indices_right]
#         keypoints_data_img_2_matched_descs = descs_img2[zero_based_indices_right]
#         keypoints_data_img_2_unmatched = np.delete(keypoints_data_img_2, zero_based_indices_right, axis=0)
#         keypoints_data_img_2_unmatched_descs = np.delete(descs_img2, zero_based_indices_right, axis=0)
#
#         # matched, img_1
#         for i in range(keypoints_data_img_1_matched.shape[0]):
#             sample = keypoints_data_img_1_matched[i,:]
#             xy = sample[0:2]
#             desc = keypoints_data_img_1_matched_descs[i,:]
#             desc_scale = sample[2]
#             desc_orientation = sample[3]
#             live_image_file_xy_green_intensity = img_1_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
#             training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
#                           (img_id_1,) + (COLMAPDatabase.array_to_blob(desc),) + (1,) + (desc_scale,) +
#                           (desc_orientation,) + (xy[0],) + (xy[1],) + (int(live_image_file_xy_green_intensity),))
#         # unmatched, img_1
#         for i in range(keypoints_data_img_1_unmatched.shape[0]):
#             sample = keypoints_data_img_1_unmatched[i, :]
#             xy = sample[0:2]
#             desc = keypoints_data_img_1_unmatched_descs[i, :]
#             desc_scale = sample[2]
#             desc_orientation = sample[3]
#             live_image_file_xy_green_intensity = img_1_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
#             training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
#                                      (img_id_1,) + (COLMAPDatabase.array_to_blob(desc),) + (0,) + (desc_scale,) + (desc_orientation,) + (xy[0],) + (xy[1],) + (
#                                      int(live_image_file_xy_green_intensity),))
#         # matched, img_2
#         for i in range(keypoints_data_img_2_matched.shape[0]):
#             sample = keypoints_data_img_2_matched[i, :]
#             xy = sample[0:2]
#             desc = keypoints_data_img_2_matched_descs[i, :]
#             desc_scale = sample[2]
#             desc_orientation = sample[3]
#             live_image_file_xy_green_intensity = img_2_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
#             training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
#                                      (img_id_2,) + (COLMAPDatabase.array_to_blob(desc),) + (1,) + (desc_scale,) + (desc_orientation,) + (xy[0],) + (xy[1],) + (
#                                      int(live_image_file_xy_green_intensity),))
#         # unmatched, img_2
#         for i in range(keypoints_data_img_2_unmatched.shape[0]):
#             sample = keypoints_data_img_2_unmatched[i, :]
#             xy = sample[0:2]
#             desc = keypoints_data_img_2_unmatched_descs[i, :]
#             desc_scale = sample[2]
#             desc_orientation = sample[3]
#             live_image_file_xy_green_intensity = img_2_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
#             training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
#                                      (img_id_2,) + (COLMAPDatabase.array_to_blob(desc),) + (0,) + (desc_scale,) + (desc_orientation,) + (xy[0],) + (xy[1],) + (
#                                      int(live_image_file_xy_green_intensity),))
#
#     print("Committing..")
#     training_data_db.commit()
#     print("Generating Stats..")
#     stats = training_data_db.execute("SELECT * FROM data").fetchall()
#     matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
#     unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()
#
#     print("Total samples: " + str(len(stats)))
#     print("Total matched samples: " + str(len(matched)))
#     print("Total unmatched samples: " + str(len(unmatched)))
#     print("% of matched samples: " + str(len(matched) * 100 / len(unmatched)))
#
#     print("Done!")
#
# base_path = sys.argv[1]
# pairs_limit = int(sys.argv[2])
# print("Base path: " + base_path)
# parameters = Parameters(base_path)
# db_live = COLMAPDatabase.connect(parameters.live_db_path)
# live_model_images = read_images_binary(parameters.live_model_images_path)
# live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)
# image_live_dir = os.path.join(base_path, 'live/images/')
# image_base_dir = os.path.join(base_path, 'base/images/')
#
# db_live_path = os.path.join(base_path, "live/database.db")
# output_path = os.path.join(base_path, "match_or_no_match_comparison_data")
# os.makedirs(output_path, exist_ok = True)
# createDataForMatchNoMatchMatchabilityComparison(image_base_dir, image_live_dir, db_live, live_model_images, output_path, pairs_limit = pairs_limit)
