import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import os
from wy_scripts.database import COLMAPDatabase


def get_rf_model(params, with_xy):
    ml_path = params.rf_model_path
    if with_xy:
        ml_path = params.rf_xy_model_path
    # ------ train the classifier model ------
    if not os.path.exists(ml_path):
        print("Training the random forest model")
        start_time = time.time()
        classify_model = train_rf_model(params.ml_db_path, with_xy)
        print("Finish training! Time: %s seconds" % (time.time() - start_time))
        joblib.dump(classify_model, ml_path)
    else:
        classify_model = joblib.load(ml_path)
    return classify_model


def train_rf_model(ml_db_path, with_xy):
    ml_database = COLMAPDatabase.connect(ml_db_path)
    # load the X, Y
    data = ml_database.execute("SELECT sift, matched FROM data").fetchall()

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    sift_vecs = sift_vecs[shuffled_idxs]
    if with_xy:
        # add the xy coordinates as additional info to train the model
        xy_cols = ml_database.execute("SELECT xy FROM data").fetchall()
        xy_coords = (COLMAPDatabase.blob_to_array(row[0], np.float64) for row in xy_cols)
        xy_coords = np.array(list(xy_coords))
        xy_coords = xy_coords[shuffled_idxs]
        sift_vecs = np.c_[sift_vecs, xy_coords]
    classes = classes[shuffled_idxs]

    clf = RandomForestClassifier(n_estimators=25, max_depth=25)
    clf.fit(sift_vecs, classes)
    return clf

