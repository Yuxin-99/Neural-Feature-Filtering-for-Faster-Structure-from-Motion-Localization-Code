import joblib
import numpy as np
import os
import random
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import time

from database import COLMAPDatabase


def get_svm_model(params, with_rgb):
    ml_path = params.svm_model_path
    # ------ train the classifier model ------
    if not os.path.exists(ml_path):
        classify_model = train_svm_model(with_rgb, params.ml_db_path, params.X_memmap_path, params.Y_memmap_path)
        joblib.dump(classify_model, ml_path)
    else:
        classify_model = joblib.load(ml_path)
    return classify_model


def train_svm_model(with_rgb, ml_db_path, X_memmap_path, Y_memmap_path):
    ml_database = COLMAPDatabase.connect(ml_db_path)
    # load the X, Y
    data = ml_database.execute("SELECT sift, matched FROM data").fetchall()

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    xy_cols = ml_database.execute("SELECT xy FROM data").fetchall()
    xy_coords = (COLMAPDatabase.blob_to_array(row[0], np.float64) for row in xy_cols)
    xy_coords = np.array(list(xy_coords))
    sift_vecs = np.c_[sift_vecs, xy_coords]
    sift_vecs = sift_vecs[shuffled_idxs]
    classes = classes[shuffled_idxs]

    # use memmap to load in the training dataset
    X_fp = np.memmap(X_memmap_path, dtype='float64', mode='w+', shape=sift_vecs.shape)
    X_fp[:] = sift_vecs[:]
    X_fp.flush()
    Y_fp = np.memmap(Y_memmap_path, dtype='int64', mode='w+', shape=classes.shape)
    Y_fp[:] = classes[:]
    Y_fp.flush()

    X_train = np.memmap(X_memmap_path, dtype='float64', shape=sift_vecs.shape)
    Y_train = np.memmap(Y_memmap_path, dtype='int64', shape=classes.shape)

    print("Training the support vector machine model")
    start_time = time.time()
    # clf = svm.SVC()
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    print("Finish training! Time: %s seconds" % (time.time() - start_time))
    return clf


def get_sgd_model(params, with_rgb):
    ml_path = params.sgd_model_path
    # ------ train the classifier model ------
    if not os.path.exists(ml_path):
        print("Training the SGD Classifier model")
        start_time = time.time()
        classify_model = train_sgd_model(params.ml_db_path, with_rgb)
        print("Finish training! Time: %s seconds" % (time.time() - start_time))
        joblib.dump(classify_model, ml_path)
    else:
        classify_model = joblib.load(ml_path)
    return classify_model


def batch(iterable_X, iterable_y, n=1):
    length = len(iterable_X)
    for ndx in range(0, length, n):
        yield iterable_X[ndx:min(ndx + n, length)], iterable_y[ndx:min(ndx + n, length)]


def train_sgd_model(ml_db_path, with_rgb):
    ml_database = COLMAPDatabase.connect(ml_db_path)
    # load the X, Y
    data = ml_database.execute("SELECT sift, matched FROM data").fetchall()

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    sift_vecs = sift_vecs[shuffled_idxs]
    xy_cols = ml_database.execute("SELECT xy FROM data").fetchall()
    xy_coords = (COLMAPDatabase.blob_to_array(row[0], np.float64) for row in xy_cols)
    xy_coords = np.array(list(xy_coords))
    xy_coords = xy_coords[shuffled_idxs]
    sift_vecs = np.c_[sift_vecs, xy_coords]
    classes = classes[shuffled_idxs]

    # print("Training dataset size: " + str(len(X_train)))

    clf = SGDClassifier(shuffle=True, loss='hinge')
    batch_iterator = batch(sift_vecs, classes, 100000)
    for index, (chunk_X, chunk_y) in enumerate(batch_iterator):
        print("SGD Classifier training turn: " + str(index))
        clf.partial_fit(chunk_X, chunk_y, classes=np.unique(chunk_y))
    return clf
