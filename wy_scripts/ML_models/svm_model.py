import joblib
import numpy as np
from sklearn import svm
import time
import os
from wy_scripts.database import COLMAPDatabase


def get_svm_model(params):
    ml_path = params.svm_model_path
    # ------ train the classifier model ------
    if not os.path.exists(ml_path):
        print("Training the support vector machine model")
        start_time = time.time()
        classify_model = train_svm_model(params.ml_db_path)
        print("Finish training! Time: %s seconds" % (time.time() - start_time))
        joblib.dump(classify_model, ml_path)
    else:
        classify_model = joblib.load(ml_path)
    return classify_model


def train_svm_model(ml_db_path):
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

    clf = svm.SVC()
    clf.fit(sift_vecs, classes)
    return clf

