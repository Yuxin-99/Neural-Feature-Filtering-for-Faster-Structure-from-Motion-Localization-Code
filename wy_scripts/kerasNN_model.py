import numpy as np
import os
import time

# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold

# import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense

from database import COLMAPDatabase


def get_kerasNN_model(params):
    ml_path = params.kerasNN_model_path
    # ------ train the classifier model ------
    if not os.path.exists(ml_path):
        print("Training the Keras Neural Network model")
        start_time = time.time()
        classify_model = train_kerasNN_model(params.ml_db_path)
        print("Finish training! Time: %s seconds" % (time.time() - start_time))
        classify_model.save(ml_path)
    else:
        classify_model = load_model(ml_path)
    return classify_model


def train_kerasNN_model(ml_db_path):
    ml_database = COLMAPDatabase.connect(ml_db_path)
    # load the X, Y training and validation data ########
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
    X = sift_vecs[shuffled_idxs]
    Y = classes[shuffled_idxs]

    train_size = int(np.ceil(0.8 * len(sift_vecs)))
    X_train = X[0:train_size]
    Y_train = Y[0:train_size]
    X_validate = X[train_size:]
    Y_validate = Y[train_size:]

    # define the Keras model ########
    model = Sequential()
    model.add(Dense(130, input_shape=(130,), activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile and train the Keras model ########
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=int(np.floor(train_size / 100)))

    # validation of the model ########
    _, accuracy = model.evaluate(X_validate, Y_validate)
    print('Accuracy: %.2f' % (accuracy * 100))
    return model
