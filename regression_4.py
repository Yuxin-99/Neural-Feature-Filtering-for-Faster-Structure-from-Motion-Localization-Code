import os
import time
import shutil

from data import getRegressionData
from tensorboard_config import get_Tensorboard_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import sys
# from sklearn.model_selection import KFold
from custom_callback import getModelCheckpointRegression, getEarlyStoppingRegression

metrics = [
      keras.metrics.MeanSquaredError(name="mean_squared_error"),
      keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
      keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error"),
      keras.metrics.CosineSimilarity(name="cosine_similarity"),
      keras.metrics.RootMeanSquaredError(name="root_mean_squared_error")
]

# sample commnad to run on bath cloud servers, ogg .. etc
# python3 regression_4.py colmap_data/Coop_data/slice1/ 32768 900 Extended_CMU_slice3 1 (or 0) (or CMU slices path)

base_path = sys.argv[1]
db_path = os.path.join(base_path, "ML_data/ml_database_all.db")
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
name = "regression_"+sys.argv[4]
train_on_matched_only = bool(int(sys.argv[5])) #this needs to be converted into an int otherwise it will read a string and return always True
score_to_train_on = sys.argv[6] #score_per_image, score_per_session, score_visibility
name = name + "_" + score_to_train_on

log_dir = get_Tensorboard_dir(name)
shutil.rmtree(log_dir)
early_stop_model_save_dir = os.path.join(log_dir, "early_stop_model")
model_save_dir = os.path.join(log_dir, "model")

print("TensorBoard log_dir: " + log_dir)
tensorboard_cb = TensorBoard(log_dir=log_dir)
print("Early_stop_model_save_dir log_dir: " + early_stop_model_save_dir)
mc_callback = getModelCheckpointRegression(early_stop_model_save_dir)
es_callback = getEarlyStoppingRegression()
all_callbacks = [tensorboard_cb, mc_callback, es_callback]

print("Running Script..!")
print(name)

print("Batch_size: " + str(batch_size))
print("Epochs: " + str(epochs))

print("Loading data..")

# minmax True returns worse results in evaluator
sift_vecs, scores = getRegressionData(db_path, minmax=False, score_name = score_to_train_on, train_on_matched_only = train_on_matched_only)

# Create model
print("Creating model")

model = Sequential()
# in keras the first layer is a hidden layer too, so input dims is OK here
model.add(Dense(128, input_dim=128, activation='relu')) #Note: 'relu' here will be the same as 'linear' (default as all SIFT values are positive)
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1)) # added relu instead of linear because all the values I expect are positive
# Compile model
opt = keras.optimizers.Adam(learning_rate=3e-4)
# The loss here will be, MeanSquaredError
model.compile(optimizer=opt, loss=keras.losses.MeanSquaredError(), metrics=metrics)
model.summary()

# Before training you should use a baseline model

# Train (or fit() )
# Just for naming's sake
X_train = sift_vecs
y_train = scores
history = model.fit(X_train, y_train,
                    validation_split=0.3,
                    epochs=epochs,
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[all_callbacks])

# Save model here
print("Saving model..")
model.save(model_save_dir)

print("Done!")