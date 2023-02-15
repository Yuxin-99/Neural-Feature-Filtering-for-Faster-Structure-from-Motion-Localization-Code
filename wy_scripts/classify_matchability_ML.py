import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from database import COLMAPDatabase
from parameters import Parameters
from read_model import read_images_binary, read_points3D_binary, load_images_from_text_file, get_localised_image_by_names, \
    get_image_id, read_image_rgb
from rf_model import get_rf_model
from svm_model import get_svm_model, get_sgd_model
from kerasNN_model import get_kerasNN_model


def main():
    base_path = sys.argv[1]
    session_id = sys.argv[2]
    parameters = Parameters(base_path, session_id, "base")
    # feature_with_xy = sys.argv[3] == "1"
    feature_with_rgb = 1

    # ------ generate the dataset for training the classifier model ------
    db_live = COLMAPDatabase.connect(parameters.live_db_path)
    live_model_images = read_images_binary(parameters.live_model_images_path)
    live_model_points3D = read_points3D_binary(parameters.live_model_points3D_path)

    create_ML_training_data(parameters.ml_db_path, live_model_points3D, live_model_images, db_live, parameters.live_img_folder)

    # ------ train the classifier model ------
    # clf_model = get_rf_model(parameters, feature_with_xy)
    clf_model = get_kerasNN_model(parameters, feature_with_rgb)
    gt_test_data = get_ml_test_data(parameters)
    test_classify_model(clf_model, gt_test_data, feature_with_rgb, parameters.kerasNN_metrics_path, parameters.slice_id)


def get_image_decs(db, image_id):
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs


def create_ML_training_data(ml_db_path, points3D, images, db, train_imgs_folder):
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
        img_name = img_data.name
        # get bgr values of the corresponding points-xy
        bgrs = read_image_rgb(img_name, train_imgs_folder, img_data.xys)
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
            bgr = bgrs[i]

            ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) +
                          (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) +
                          (COLMAPDatabase.array_to_blob(bgr),) + (matched,))
        img_index += 1

    print()
    print('Done!')
    ml_db.commit()

    print("Generating Data Info...")
    all_data = ml_db.execute("SELECT * FROM data ORDER BY image_id DESC").fetchall()

    all_sifts = (COLMAPDatabase.blob_to_array(row[2], np.uint8) for row in all_data)
    all_sifts = np.array(list(all_sifts))

    all_classes = (row[6] for row in all_data)  # binary values
    all_classes = np.array(list(all_classes))

    print(" Total Training Size: " + str(all_sifts.shape[0]))
    ratio = np.where(all_classes == 1)[0].shape[0] / np.where(all_classes == 0)[0].shape[0]
    print("Ratio of Positives to Negatives Classes: " + str(ratio))


def get_ml_test_data(params):
    # get test data too (gt = query as we know)
    db_gt = COLMAPDatabase.connect(params.query_db_path)

    query_images_names = load_images_from_text_file(params.query_images_path)
    localised_query_images_names = get_localised_image_by_names(query_images_names, params.query_gt_img_bin_path)

    gt_points_3D = read_points3D_binary(params.query_points_bin_path)
    gt_model_images = read_images_binary(params.query_gt_img_bin_path)

    data_to_write = np.empty([0, 134])
    for loc_img_name in tqdm(localised_query_images_names):
        image_id = get_image_id(db_gt, loc_img_name)
        image = gt_model_images[int(image_id)]
        kp_db_row = db_gt.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(image_id) + "'").fetchone()
        rows = kp_db_row[0]
        descs = get_image_decs(db_gt, image_id)
        assert (image.xys.shape[0] == image.point3D_ids.shape[0] == rows == descs.shape[0])  # just for my sanity
        # get bgr values of the corresponding points-xy
        bgrs = read_image_rgb(loc_img_name, params.query_img_folder, image.xys)

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
        data = np.c_[descs, image.xys]
        data = np.c_[data, bgrs]
        data = np.c_[data, matched_values]
        data_to_write = np.r_[data_to_write, data]
    return data_to_write.astype(np.float32)


def test_classify_model(clf_model, gt_test_data, with_rgb, ml_metrics_path, slice_id):
    # ------ test and evaluate the classifier model ------
    print("Testing the model")
    if with_rgb:
        X_test = gt_test_data[:, 0:133]
        y_true = gt_test_data[:, 133].astype(np.uint8)
    else:
        X_test = gt_test_data[:, 0:130]
        y_true = gt_test_data[:, 130].astype(np.uint8)
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
    with open(ml_metrics_path, 'w') as f:
        f.write("Data slice: " + slice_id)
        f.write('\n')
        f.write("Precision score: " + str(precision))
        f.write('\n')
        f.write("Recall score: " + str(recall))
        f.write('\n')
        f.write("Specificity score: " + str(specificity))
        f.write('\n')
        f.write("Accuracy score: " + str(accuracy))
        f.write('\n')


if __name__ == "__main__":
    main()
