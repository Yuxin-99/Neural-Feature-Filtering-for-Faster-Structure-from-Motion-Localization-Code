import numpy as np
import os
import time

# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold

# import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import Sequence
from sklearn import preprocessing
from tensorflow import keras

from database import COLMAPDatabase
from MSFE_loss_NN import tweaked_loss


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def get_kerasNN_model(params, feature_with_rgb, use_MSFE):
    ml_path = params.kerasNN_model_path
    if feature_with_rgb:
        ml_path = params.kerasNN_rgb_path
    if use_MSFE:
        ml_path = params.MSFENN_path
        if feature_with_rgb:
            ml_path = params.MSFENN_rgb_path
    # ------ train the classifier model ------
    if not os.path.exists(ml_path):
        print("Training the Keras Neural Network model")
        start_time = time.time()
        classify_model = train_kerasNN_model(params.ml_db_path, feature_with_rgb, use_MSFE)
        print("Finish training! Time: %s seconds" % (time.time() - start_time))
        classify_model.save(ml_path)
    else:
        if use_MSFE:
            classify_model = load_model(ml_path, compile=False)
        else:
            classify_model = load_model(ml_path)
    return classify_model


def train_kerasNN_model(ml_db_path, feature_with_rgb, use_MSFE):
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
    if feature_with_rgb:
        bgr_cols = ml_database.execute("SELECT rgb FROM data").fetchall()
        bgrs = (COLMAPDatabase.blob_to_array(row[0], np.float64) for row in bgr_cols)
        bgrs = np.array(list(bgrs))
        sift_vecs = np.c_[sift_vecs, bgrs]

    X = sift_vecs[shuffled_idxs]
    # X_normalized = preprocessing.normalize(X, axis=0)
    Y = classes[shuffled_idxs]

    train_size = int(np.ceil(0.8 * len(sift_vecs)))
    X_train = X[0:train_size]
    Y_train = Y[0:train_size]
    X_validate = X[train_size:]
    Y_validate = Y[train_size:]

    # define the Keras model ########
    feature_num = X_train.shape[1]
    model = Sequential()
    model.add(Dense(feature_num, input_dim=feature_num, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(feature_num, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    if use_MSFE:
        model.compile(optimizer=opt, loss=tweaked_loss)
    else:
        model.compile(optimizer=opt, loss='binary_crossentropy')
        # model.add(Dense(feature_num, input_shape=(feature_num,), activation='relu'))
        # model.add(Dense(60, activation='relu'))
        # model.add(Dense(30, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # # compile and train the Keras model ########
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # epochs = 100
        # batch_size = int(np.floor(train_size / 100))
        #
        # model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
        # # validation of the model ########
        # _, accuracy = model.evaluate(X_validate, Y_validate)
        # print('Accuracy: %.2f' % (accuracy * 100))

    model.summary()
    epochs = 250
    batch_size = 4096
    Y_train = Y_train.astype(np.float32)
    Y_validate = Y_validate.astype(np.float32)

    training_gen_data = DataGenerator(X_train, Y_train, batch_size)
    validation_gen_data = DataGenerator(X_validate, Y_validate, batch_size)

    model.fit(training_gen_data, validation_data=validation_gen_data, epochs=epochs, shuffle=True, verbose=1)

    return model
