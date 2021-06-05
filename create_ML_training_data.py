import pdb
import sys
from collections import defaultdict
import numpy as np
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary
from sklearn.model_selection import train_test_split

# The ML part for my second publication starts from here:
# This file will use data from previous publication (live model + all sessions) and the live database. You can run this on your alienware or ogg/weatherwax
# You run these in order:
# (Note, load the python venv: source venv/bin/activate (not in docker!))
# run, create_ML_training_data.py (see below)
# then run any model such as regression.py, regression_rf.py, using docker on weatherwax or ogg cs.bath.ac.uk.
# (Note the docker command to run is (and before run this "hare reserve 20000" to reserve a port , change to "fullpipeline dir first" ):
# hare run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 -v "$(pwd)":/fullpipeline --workdir /fullpipeline -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -p 20000:80 -ti bath:2020-gpu (the old interactive command)
# (Note, you will need docker to run the models because it uses gpus, the venv uses python3.6 for some reason - it's ok I think)

# for docker you might also need to run these for "cv2" and "cvxpnpl" (or add to Dockerfiles)
# might need this too: pip install tensorflow (not for bath:2020-gpu image)
# apt-get update && apt-get install ffmpeg libsm6 libxext6 libblas-dev liblapack-dev -y && pip install opencv-contrib-python && pip install scs && pip install cvxpnpl

# Tensorboard Notes:
# https://chadrick-kwag.net/how-to-manually-write-to-tensorboard-from-tf-keras-callback-useful-trick-when-writing-a-handful-of-validation-metrics-at-once/
# You need 2 terminals
# 1 - to run tensorboard, you ssh with "ssh -L 9999:localhost:20000 ar2056@weatherwax.cs.bath.ac.uk", and run this too "source ~/venv_basic/bin/activate"
# then run "tensorboard --logdir colmap_data/Coop_data/slice1/ML_data/results/ --port 20000" inside fullpipeline/ (might need to reserve a port with hare)
# the you visit "http://localhost:9999" on your local machine.
# 2 - the terminal you usually run the hare command from above and Tensorflow will read from the dir you store the results.
# Will need to flush with the writer though, https://stackoverflow.com/questions/52483296/when-do-i-have-to-use-tensorflows-filewriter-flush-method (doesn't work...)

# Docker Notes:
# create an image using the docker file under "/homes/ar2056/docker"
# hare build -t ar2056/basic . (whatever name you prefer)
# hare run -dit --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --workdir /home -v /homes/ar2056/:/home/ --name ar2056_basic ar2056/basic:latest (build container)
# run this first (from laptop), then launch Pycharm for remote dev
# ssh -L 6000:172.17.0.5:22 ar2056@weatherwax.cs.bath.ac.uk (make sure the IP points to a full-working docker, and the docker has to have the ip of 172.17.0.5 or same)
# use git locally on your laptop - the cloud does not like git
# get IP of container with "hare inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container name>"

# Mine not needed anymore
# def split_data(features, target, test_percentage, randomize = False):
#     if(randomize):
#         print("Randomizing data")
#         union = np.c_[features, target]
#         np.random.shuffle(union)
#         features = union[:, 0:128]
#         target = union[:, 128]
#     rows_no = features.shape[0] #or test , same thing
#     train_percentage = 1 - test_percentage
#     train_max_idx = int(np.floor(rows_no * train_percentage))
#     X_train = features[0 :  train_max_idx , :]
#     y_train = target[0 : train_max_idx]
#     X_test = features[train_max_idx : , :]
#     y_test = target[train_max_idx :]
#     return X_train, y_train, X_test, y_test

def get_image_decs(db, image_id): #not to be confused with get_queryDescriptors() in feature_matching_generator.py - that one normalises descriptors.
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs

def get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index):
    point_index = points3D_id_index[current_point3D_id]
    point_score = points3D_scores[point_index]
    return point_score

def create_all_data(ml_db_path, points3D, points3D_id_index, points3D_reliability_scores, points3D_heatmap_vals, points3D_visibility_vals, images, db):
    ml_db = COLMAPDatabase.create_db_for_all_data(ml_db_path) #returns a connection
    img_index = 0
    ml_db.execute("BEGIN")
    for img_id , img_data in images.items():
        print("Doing image " + str(img_index + 1) + "/" + str(len(images.items())), end="\r")
        descs = get_image_decs(db, img_id)
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0]) # just for my sanity
        for i in range(img_data.point3D_ids.shape[0]): # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if(current_point3D_id == -1): # means feature is unmatched
                per_image_score = 0
                per_session_score = 0
                visibility_score = 0
                matched = 0
                xyz = np.array([0, 0, 0])  # safe to use as no image point will ever match to 0,0,0
            else:
                per_image_score = get_point3D_score(points3D_reliability_scores, current_point3D_id, points3D_id_index)
                per_session_score = get_point3D_score(points3D_heatmap_vals, current_point3D_id, points3D_id_index)
                visibility_score = get_point3D_score(points3D_visibility_vals, current_point3D_id, points3D_id_index)
                matched = 1
                xyz = points3D[current_point3D_id].xyz  # np.float64

            desc = descs[i] # np.uint8
            xy = img_data.xys[i] #np.float64, same as xyz
            img_name = img_data.name

            ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) + (per_image_score,) + (per_session_score,) + (visibility_score,) + (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) + (matched,))
        img_index +=1

    print()
    print('Done!')
    ml_db.commit()

    print("Generating Data Info...")
    all_data = ml_db.execute("SELECT * FROM data ORDER BY image_id DESC").fetchall()

    all_sifts = (COLMAPDatabase.blob_to_array(row[2], np.uint8) for row in all_data)
    all_sifts = np.array(list(all_sifts))

    per_image_scores = (row[3] for row in all_data)  # continuous values
    per_image_scores = np.array(list(per_image_scores))

    per_session_scores = (row[4] for row in all_data)  # continuous values
    per_session_scores = np.array(list(per_session_scores))

    visibility_scores = (row[5] for row in all_data)  # continuous values
    visibility_scores = np.array(list(visibility_scores))

    all_classes = (row[8] for row in all_data)  # binary values
    all_classes = np.array(list(all_classes))

    print(" Total Training Size: " + str(all_sifts.shape[0]))
    print(" per_image_scores mean: " + str(per_image_scores.mean()))
    print(" per_session_scores mean: " + str(per_session_scores.mean()))
    print(" visibility_scores mean: " + str(visibility_scores.mean()))
    ratio = np.where(all_classes == 1)[0].shape[0] / np.where(all_classes == 0)[0].shape[0]
    print("Ratio of Positives to Negatives Classes: " + str(ratio))

base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice1/" #trailing "/"
parameters = Parameters(base_path)

db_live = COLMAPDatabase.connect(parameters.live_db_path)

live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)

# Getting the scores
points3D_reliability_scores_matrix= np.load(parameters.per_image_decay_matrix_path)
points3D_heatmap_vals_matrix = np.load(parameters.per_session_decay_matrix_path)
points3D_visibility_matrix = np.load(parameters.binary_visibility_matrix_path)

points3D_reliability_scores = points3D_reliability_scores_matrix.sum(axis=0)
points3D_heatmap_vals = points3D_heatmap_vals_matrix.sum(axis=0)
points3D_visibility_vals = points3D_visibility_matrix.sum(axis=0)

points3D_id_index = index_dict_reverse(live_model_points3D)

# i.e /home/alex/fullpipeline/colmap_data/alfa_mega/slice1/ML_data/database.db / or ml_database.db / or coop/alfa_mega
# make sure you delete the databases (.db) file first!
ml_db_path = sys.argv[2] #colmap_data/Coop_data/slice1/ML_data/ml_database_all.db
db_path_train = sys.argv[3] #colmap_data/Coop_data/slice1/ML_data/ml_database_train.db

print("Creating all training data..")
# this was create to simplify process, create a db with all the data then create a test and train database (as of 04/05/2021, test db is not used)
create_all_data(ml_db_path, live_model_points3D, points3D_id_index,
                points3D_reliability_scores, points3D_heatmap_vals, points3D_visibility_vals,
                live_model_images, db_live)
