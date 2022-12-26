import numpy as np
import joblib
import os
import sys
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from database import COLMAPDatabase
from parameters import Parameters
from read_model import read_images_binary, read_points3D_binary, load_images_from_text_file, get_localised_image_by_names, get_image_id


def main():
    base_path = sys.argv[1]
    session_id = sys.argv[2]
    parameters = Parameters(base_path, session_id)

    # ------ generate the dataset for training the classifier model ------
    db_live = COLMAPDatabase.connect(parameters.live_db_path)
    live_model_images = read_images_binary(parameters.live_model_images_path)
    live_model_points3D = read_points3D_binary(parameters.live_model_points3D_path)

    create_ML_training_data(parameters.ml_db_path, live_model_points3D, live_model_images, db_live)

    # ------ train the classifier model ------
    # clf_model = get_trained_classify_model(parameters)
    # test_classify_model(clf_model, parameters)


def get_trained_classify_model(parameters):
    # ------ train the classifier model ------
    if not os.path.exists(parameters.clf_ml_path):
        print("Training the model")
        start_time = time.time()
        classify_model = train_classify_model(parameters.ml_db_path)
        print("Finish training! Time: %s seconds" % (time.time() - start_time))
        joblib.dump(classify_model, parameters.clf_ml_path)
    else:
        classify_model = joblib.load(parameters.clf_ml_path)
    return classify_model


def get_image_decs(db, image_id):
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs


def create_ML_training_data(ml_db_path, points3D, images, db):
    if os.path.exists(ml_db_path):
        print("ml database already exists.")
        return

    print("Creating all training data..")
    # this was created to simplify process, create a db with all the data then create a test and train database
    ml_db = COLMAPDatabase.create_db_for_all_data(ml_db_path)
    img_index = 0
    ml_db.execute("BEGIN")
    for img_id, img_data in images.items():
        print("Doing image " + str(img_index + 1) + "/" + str(len(images.items())), end="\r")
        descs = get_image_decs(db, img_id)
        assert (img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0])  # just for my sanity
        for i in range(img_data.point3D_ids.shape[0]):  # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if current_point3D_id == -1:  # means feature is unmatched
                matched = 0
                xyz = np.array([0, 0, 0])  # safe to use as no image point will ever match to 0,0,0
            else:
                matched = 1
                xyz = points3D[current_point3D_id].xyz  # np.float64

            desc = descs[i]  # np.uint8
            xy = img_data.xys[i]  # np.float64, same as xyz
            img_name = img_data.name

            ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?)",
                          (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) +
                          (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) + (matched,))
        img_index += 1

    print()
    print('Done!')
    ml_db.commit()

    print("Generating Data Info...")
    all_data = ml_db.execute("SELECT * FROM data ORDER BY image_id DESC").fetchall()

    all_sifts = (COLMAPDatabase.blob_to_array(row[2], np.uint8) for row in all_data)
    all_sifts = np.array(list(all_sifts))

    all_classes = (row[5] for row in all_data)  # binary values
    all_classes = np.array(list(all_classes))

    print(" Total Training Size: " + str(all_sifts.shape[0]))
    ratio = np.where(all_classes == 1)[0].shape[0] / np.where(all_classes == 0)[0].shape[0]
    print("Ratio of Positives to Negatives Classes: " + str(ratio))


def train_classify_model(ml_db_path):
    ml_database = COLMAPDatabase.connect(ml_db_path)
    # load the X, Y
    data = ml_database.execute("SELECT sift, matched FROM data").fetchall()

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    sift_vecs = sift_vecs[shuffled_idxs]
    classes = classes[shuffled_idxs]
    clf = RandomForestClassifier(n_estimators=25, max_depth=25)

    clf.fit(sift_vecs, classes)
    return clf


def test_classify_model(clf_model, params):
    # ------ test and evaluate the classifier model ------
    print("Testing the model")
    gt_test_data = get_ml_test_data(params)
    X_test = gt_test_data[:, 0:128]
    y_true = gt_test_data[:, 128].astype(np.uint8)
    y_pred_pos = clf_model.predict(X_test)
    y_pred_class = y_pred_pos > 0.5

    print("Evaluating the model")
    cm = confusion_matrix(y_true, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    # how many observations predicted as positive are in fact positive.
    precision = precision_score(y_true, y_pred_class)  # or optionally tp/ (tp + fp)
    print("Precision score: " + str(precision))
    # true positive rate: how many observations out of all positive observations have we classified as positive
    recall = recall_score(y_true, y_pred_class)  # or optionally tp / (tp + fn)
    print("Recall score: " + str(recall))
    # true negative rate: how many observations out of all negative observations have we classified as negative
    specificity = tn / (tn + fp)
    print("Specificity score: " + str(specificity))
    # how many observations, both positive and negative, were correctly classified.
    # shouldn't be used on imbalanced problem
    accuracy = accuracy_score(y_true, y_pred_class)  # or optionally (tp + tn) / (tp + fp + fn + tn)
    print("Accuracy score: " + str(accuracy))
    with open(params.clf_ml_metrics_path, 'w') as f:
        f.write("Precision score: " + str(precision))
        f.write('\n')
        f.write("Recall score: " + str(recall))
        f.write('\n')
        f.write("Specificity score: " + str(specificity))
        f.write('\n')
        f.write("Accuracy score: " + str(accuracy))
        f.write('\n')


def get_ml_test_data(params):
    # get test data too (gt = query as we know)
    db_gt = COLMAPDatabase.connect(params.query_db_path)

    query_images_names = load_images_from_text_file(params.query_images_path)
    localised_query_images_names = get_localised_image_by_names(query_images_names, params.query_gt_img_bin_path)

    gt_points_3D = read_points3D_binary(params.query_points_bin_path)
    gt_model_images = read_images_binary(params.query_gt_img_bin_path)

    data_to_write = np.empty([0, 129])
    for loc_img_name in tqdm(localised_query_images_names):
        image_id = get_image_id(db_gt, loc_img_name)
        image = gt_model_images[int(image_id)]
        kp_db_row = db_gt.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(image_id) + "'").fetchone()
        rows = kp_db_row[0]
        descs = get_image_decs(db_gt, image_id)
        assert (image.xys.shape[0] == image.point3D_ids.shape[0] == rows == descs.shape[0])  # just for my sanity

        matched_values = []  # for each keypoint (x,y)/desc same thing
        for i in range(image.xys.shape[0]):  # can loop through descs or img_data.xys - same order
            current_point3D_id = image.point3D_ids[i]
            if current_point3D_id == -1:  # means feature is unmatched
                matched = 0
            else:
                # this is to make sure that xy belong to the right pointd3D
                assert i in gt_points_3D[current_point3D_id].point2D_idxs
                matched = 1
            matched_values.append(matched)

        matched_values = np.array(matched_values).reshape(rows, 1)
        data = np.c_[descs, matched_values]
        data_to_write = np.r_[data_to_write, data]
    return data_to_write.astype(np.float32)


if __name__ == "__main__":
    main()
