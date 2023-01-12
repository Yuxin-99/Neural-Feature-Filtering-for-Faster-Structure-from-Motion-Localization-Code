import joblib
import numpy as np
import os
from sklearn import svm
import time

from database import COLMAPDatabase


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

    train_size = int(np.ceil(0.2 * len(sift_vecs)))
    X_train = sift_vecs[0:train_size]
    Y_train = classes[0:train_size]
    print("Training dataset size: " + str(len(X_train)))

    clf = svm.SVC()
    # clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    return clf
