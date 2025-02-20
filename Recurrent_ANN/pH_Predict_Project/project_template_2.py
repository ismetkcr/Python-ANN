# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:39:24 2025

@author: ismt
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:23:12 2025

@author: ismt
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_pH_RNN_project import read_excel_file_and_plot, normalize_dataframe, make_multivariate_windows, \
predict_1d, make_train_test_splits, predict_2d, NBeatsBlock, predict_ensemble
import random
import os
# Set all random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
from tensorflow.keras.callbacks import EarlyStopping
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


import os
import tensorflow as tf

def get_model_checkpoint_callback(model_name, monitor="val_accuracy", mode="max", save_best_only=True):
    """
    Creates a ModelCheckpoint callback that saves the best model inside 'model_experiments' directory.

    Parameters:
    - model_name (str): Name of the model, used for the filename.
    - monitor (str): Metric to monitor (e.g., "val_loss", "val_accuracy").
    - mode (str): "max" for accuracy, "min" for loss.
    - save_best_only (bool): Whether to save only the best model.

    Returns:
    - tf.keras.callbacks.ModelCheckpoint: The callback object.
    """
    # Create directory if it doesn't exist
    save_dir = "model_experiments"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the file path
    checkpoint_filepath = os.path.join(save_dir, f"{model_name}.keras")
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=False,
        verbose=1
    )
    
    return checkpoint_callback


class SaveLastEpochModel(tf.keras.callbacks.Callback):
    def __init__(self, save_path, model_name):
        super().__init__()
        self.save_path = save_path
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params["epochs"] - 1:  # Check if it's the last epoch
            self.model.save(os.path.join(self.save_path, self.model_name))
            

from tensorflow.keras.callbacks import Callback

class VerboseEveryNEpochs(Callback):
    def __init__(self, N):
        self.N = N  # Print every N epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.N == 0:  # Check if it's a multiple of N
            logs = logs or {}
            print(f"Epoch {epoch+1}: " + ", ".join([f"{key}={value:.4f}" for key, value in logs.items()]))

verbose_callback = VerboseEveryNEpochs(5)


# #read data from experiment_values excel file
# square_df = read_excel_file_and_plot("experiment_values_square_150.xlsx", show=False)
# prbs4_df = read_excel_file_and_plot("experiment_values_prbs4_150.xlsx", show=False)
# prbs7_df = read_excel_file_and_plot("experiment_values_prbs7_150.xlsx", show=True)
# random_df = read_excel_file_and_plot("experiment_values_random_vals.xlsx", show=False)
train_df = read_excel_file_and_plot("train_data_pH.xlsx", show=False)
validation_df = read_excel_file_and_plot("test_data_pH.xlsx", show=False)


train_df_normalized = normalize_dataframe(train_df, show=False)
validation_df_normalized = normalize_dataframe(validation_df, show=False)


# # Normalize the DataFrame and plot the normalized values
# square_normalized_df = normalize_dataframe(square_df, show=False)
# prbs4_normalized_df = normalize_dataframe(prbs4_df, show=False)
# prbs7_normalized_df = normalize_dataframe(prbs7_df, show=False)
# random_normalized_df = normalize_dataframe(random_df, show=False)

# #make windows of data 
# WINDOW_SIZE=10
# BATCH_SIZE=32


# all_windows_arx, all_labels_arx = make_multivariate_windows(data=train_df_normalized, window_size=WINDOW_SIZE)
# validation_data_windows, validation_data_labels = make_multivariate_windows(data=validation_df_normalized, window_size=WINDOW_SIZE)


# #Make data 1D for ARX Model
# windows_arx = []
# for window in all_windows_arx:
#     window_arx = window.T.reshape(WINDOW_SIZE*2,) #window_size*2
#     windows_arx.append(window_arx)

# windows_arx = np.array(windows_arx)
# print("Windows arx shape", all_windows_arx.shape)
# print("labels shape:", all_labels_arx.shape)
# # print("First window:", all_windows_arx[0])
# # print("First label:", all_labels_arx[0])       

# validation_data_windows_arx = []
# for window in validation_data_windows:
#     window_arx = window.T.reshape(WINDOW_SIZE*2,) #window_size*2
#     validation_data_windows_arx.append(window_arx)
    
    
# validation_data_windows_arx = np.array(validation_data_windows_arx)
# print("validation Windows arx shape", validation_data_windows_arx.shape)
# print("validation labels shape:", validation_data_labels.shape)    
    



# #THIS IS FOR ARX Model 1D data #be aware of that: test labels and test u must have to come from same dataset fot this example its random u values dataset
# #length of test labels cannot exceed last dataset length
# train_windows_arx, test_windows_arx, train_labels , test_labels = make_train_test_splits(windows_arx, all_labels_arx, test_split=0.11)
# print(len(train_windows_arx), len(test_windows_arx), len(train_labels), len(test_labels))

# train_features_dataset_arx = tf.data.Dataset.from_tensor_slices(train_windows_arx)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)

# test_features_dataset_arx = tf.data.Dataset.from_tensor_slices(test_windows_arx)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)

# train_dataset_arx=tf.data.Dataset.zip((train_features_dataset_arx, train_labels_dataset))
# test_dataset_arx=tf.data.Dataset.zip((test_features_dataset_arx, test_labels_dataset))
# #batch and prefetch
# train_dataset_arx=train_dataset_arx.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_arx=test_dataset_arx.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




# inputs=Input(shape=windows_arx.shape[1],name="input_layer") #T,
# outputs=Dense(1, name="output_layer")(inputs)
# model_1=tf.keras.models.Model(inputs=inputs, outputs=outputs, name="ARX_Model")

# model_1.compile(loss="mae",
#               optimizer=Adam())


# history_ARX = model_1.fit(train_dataset_arx,
#                         epochs=25,
#                         verbose=1,
#                         validation_data=test_dataset_arx,
#                         callbacks=[early_stopping])                        

# loss_arx=history_ARX.history["val_loss"]


# plt.figure()
# plt.plot(loss_arx, label="arxmodel_Val_loss")
# plt.legend()
# plt.show()
# model_1.summary()

# #Test data prediction
# prediction_horizon = len(test_labels) 
# arx_predictions, arx_results = predict_1d(
#     model=model_1,
#     test_windows_1d=test_windows_arx,
#     test_labels=test_labels,
#     test_u=train_df_normalized['base_flow'].values[-len(test_labels):],
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon
# )

# print("Evaluation Metrics:", arx_results)

# #Validation_data_prediction
# prediction_horizon = 100
# arx_predictions, arx_results = predict_1d(
#     model=model_1,
#     test_windows_1d=validation_data_windows_arx,
#     test_labels=validation_data_labels,
#     test_u=validation_df_normalized['base_flow'].values,
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
#     start_index = 500
# )

# print("Evaluation Metrics:", arx_results)

#ANN windowed data 
# WINDOW_SIZE=30
# BATCH_SIZE=4


# all_windows_ann, all_labels_ann = make_multivariate_windows(data=train_df_normalized, window_size=WINDOW_SIZE)
# validation_data_windows, validation_data_labels = make_multivariate_windows(data=validation_df_normalized, window_size=WINDOW_SIZE)



# # windows_ann_square, labels_ann_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# # print("ann Windows shape", windows_ann_square.shape)
# # print("ann labels shape:", labels_ann_square.shape)


# # windows_ann_prbs4, labels_ann_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# # print("prbs4 Windows shape", windows_ann_prbs4.shape)
# # print("prbs4 labels shape:", labels_ann_prbs4.shape)


# # windows_ann_prbs7, labels_ann_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# # print("prbs7 Windows shape", windows_ann_prbs7.shape)
# # print("prbs7 labels shape:", labels_ann_prbs7.shape)


# # windows_ann_random, labels_ann_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# # print("random Windows shape", windows_ann_random.shape)
# # print("random labels shape:", labels_ann_random.shape)


# # all_windows_ann = np.concatenate([windows_ann_square, windows_ann_prbs4, windows_ann_prbs7, windows_ann_random])
# # all_labels_ann = np.concatenate([labels_ann_square, labels_ann_prbs4, labels_ann_prbs7, labels_ann_random])

# # all_windows = np.concatenate([windows_prbs7, windows_random])
# # all_labels = np.concatenate([labels_prbs7, labels_random])
# print(len(all_windows_ann), len(all_labels_ann))


# #Make data 1D for ANN Model
# windows_ann = []
# for window in all_windows_ann:
#     window_ann = window.T.reshape(WINDOW_SIZE*2,) #window_size*2
#     windows_ann.append(window_ann)

# windows_ann = np.array(windows_ann)
# print("Windows ann shape", all_windows_ann.shape)
# print("labels shape:", all_labels_ann.shape)
# # print("First window:", all_windows_ann[0])
# # print("First label:", all_labels_ann[0])       

# validation_data_windows_arx = []
# for window in validation_data_windows:
#     window_arx = window.T.reshape(WINDOW_SIZE*2,) #window_size*2
#     validation_data_windows_arx.append(window_arx)

# #THIS IS FOR ANN Model 1D data 
# train_windows_ann, test_windows_ann, train_labels , test_labels = make_train_test_splits(windows_ann, all_labels_ann, test_split=0.025)
# print(len(train_windows_ann), len(test_windows_ann), len(train_labels), len(test_labels))

# train_features_dataset_ann = tf.data.Dataset.from_tensor_slices(train_windows_ann)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)

# test_features_dataset_ann = tf.data.Dataset.from_tensor_slices(test_windows_ann)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)

# train_dataset_ann=tf.data.Dataset.zip((train_features_dataset_ann, train_labels_dataset))
# test_dataset_ann=tf.data.Dataset.zip((test_features_dataset_ann, test_labels_dataset))
# #batch and prefetch
# train_dataset_ann=train_dataset_ann.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_ann=test_dataset_ann.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# #Model 2. Simple ANN Model
# inputs = tf.keras.layers.Input(shape=windows_ann.shape[1],) #same data format as arx model
# x = tf.keras.layers.Dense(1024, activation="relu")(inputs)
# x = tf.keras.layers.Dense(1024, activation="relu")(x)
# outputs = tf.keras.layers.Dense(1, activation="linear")(x)
# model_2 = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="SimpleANN")

# model_2.compile(loss="mae",
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))


# # Train ANN model with early stopping
# history_ann = model_2.fit(train_dataset_ann,
#             epochs=500,
#             validation_data=test_dataset_ann,
#             verbose=0,
#             callbacks=[early_stopping, verbose_callback])


# loss_ann=history_ann.history["val_loss"]
# plt.figure()
# plt.plot(loss_ann, label="annmodelvalloss")
# plt.legend()
# plt.show()
# model_2.summary()
# # Example call
# #datanın sonundan itibaren predict ediyor isem -predict_horizon: calısır.
# prediction_horizon = len(test_labels) 
# ann_predictions, ann_results = predict_1d(
#     model=model_2,
#     test_windows_1d=test_windows_ann,
#     test_labels=test_labels,
#     test_u=train_df_normalized['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
# )
# #datanın basından itibaren predict ediyor isem WINDOW_SIZE: olarak vermeliyim.. mantıkan predict horizon 400 olsa son dataları içermeyecek...
# #Validation_data_prediction
# prediction_horizon = 400
# ann_predictions, ann_results = predict_1d(
#     model=model_2,
#     test_windows_1d=validation_data_windows_arx,
#     test_labels=validation_data_labels,
#     test_u=validation_df_normalized['base_flow'].values[WINDOW_SIZE:],
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
#     start_index = 100
# )

# # # #this is for RNN and CNN MODEL DATA expects data T, D

# # CNN windowed data 
# WINDOW_SIZE=30
# BATCH_SIZE=4
# HORIZON=1

# all_windows_cnn, all_labels_cnn = make_multivariate_windows(data=train_df_normalized, window_size=WINDOW_SIZE)
# validation_data_windows, validation_data_labels = make_multivariate_windows(data=validation_df_normalized, window_size=WINDOW_SIZE)


# # windows_cnn_square, labels_cnn_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# # print("cnn Windows shape", windows_cnn_square.shape)
# # print("cnn labels shape:", labels_cnn_square.shape)


# # windows_cnn_prbs4, labels_cnn_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# # print("prbs4 Windows shape", windows_cnn_prbs4.shape)
# # print("prbs4 labels shape:", labels_cnn_prbs4.shape)


# # windows_cnn_prbs7, labels_cnn_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# # print("prbs7 Windows shape", windows_cnn_prbs7.shape)
# # print("prbs7 labels shape:", labels_cnn_prbs7.shape)


# # windows_cnn_random, labels_cnn_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# # print("random Windows shape", windows_cnn_random.shape)
# # print("random labels shape:", labels_cnn_random.shape)


# # all_windows_cnn = np.concatenate([windows_cnn_square, windows_cnn_prbs4, windows_cnn_prbs7, windows_cnn_random])
# # all_labels_cnn = np.concatenate([labels_cnn_square, labels_cnn_prbs4, labels_cnn_prbs7, labels_cnn_random])
# # print(len(all_windows_cnn), len(all_labels_cnn))

# print("Windows cnn shape", all_windows_cnn.shape)
# print("labels shape:", all_labels_cnn.shape)
# # print("First window:", all_windows_cnn[0])
# # print("First label:", all_labels_cnn[0])    


# train_windows_cnn, test_windows_cnn, train_labels , test_labels = make_train_test_splits(all_windows_cnn, all_labels_cnn, test_split=0.025)
# len(train_windows_cnn), len(test_windows_cnn), len(train_labels), len(test_labels)

# train_features_dataset_cnn = tf.data.Dataset.from_tensor_slices(train_windows_cnn)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)


# test_features_dataset_cnn = tf.data.Dataset.from_tensor_slices(test_windows_cnn)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)


# train_dataset_cnn=tf.data.Dataset.zip((train_features_dataset_cnn, train_labels_dataset))
# test_dataset_cnn=tf.data.Dataset.zip((test_features_dataset_cnn, test_labels_dataset))

# train_dataset_cnn=train_dataset_cnn.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_cnn=test_dataset_cnn.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# inputs = tf.keras.layers.Input(shape=(train_windows_cnn[0].shape))
# x = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding="causal", activation="relu")(inputs)
# x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding="causal", activation="relu")(x)
# x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding="causal", activation="relu")(x)
# x = tf.keras.layers.GlobalAveragePooling1D()(x)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
# outputs = tf.keras.layers.Dense(HORIZON)(x)
# model_3 = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="cnn_model")

# # Compile with learning rate adjustment
# model_3.compile(loss="mae",
#                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
#                )

# history_cnn = model_3.fit(train_dataset_cnn,
#             epochs=50,
#             verbose=0,
#             validation_data=test_dataset_cnn,
#             callbacks=[early_stopping, verbose_callback])

# model_3.summary()
# loss_cnn = history_cnn.history["val_loss"]
# plt.figure()
# plt.plot(loss_cnn, label="cnnmodelvalloss")
# plt.legend()
# plt.show()

# # Example call
# prediction_horizon = len(test_labels[:300]) 
# cnn_predictions, cnn_results = predict_2d(
#     model=model_3,
#     test_windows_2d=test_windows_cnn,
#     test_labels=test_labels,
#     test_u=train_df_normalized['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
#     start_index=0
# )

# print("Evaluation Metrics:", cnn_results)

# #Validation_data_prediction
# prediction_horizon = 400
# ann_predictions, ann_results = predict_2d(
#     model=model_3,
#     test_windows_2d=validation_data_windows,
#     test_labels=validation_data_labels,
#     test_u=validation_df_normalized['base_flow'].values[WINDOW_SIZE:],
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
#     start_index = 100
# )


##Dataset For RNN Models: SimpleRNN, GRU, LSTM 
##RNN windowed data 

WINDOW_SIZE=30
BATCH_SIZE=2
HORIZON=1

all_windows_rnn, all_labels_rnn = make_multivariate_windows(data=train_df_normalized, window_size=WINDOW_SIZE)
validation_data_windows, validation_data_labels = make_multivariate_windows(data=validation_df_normalized, window_size=WINDOW_SIZE)


# windows_rnn_square, labels_rnn_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# print("rnn Windows shape", windows_rnn_square.shape)
# print("rnn labels shape:", labels_rnn_square.shape)


# windows_rnn_prbs4, labels_rnn_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# print("prbs4 Windows shape", windows_rnn_prbs4.shape)
# print("prbs4 labels shape:", labels_rnn_prbs4.shape)


# windows_rnn_prbs7, labels_rnn_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# print("prbs7 Windows shape", windows_rnn_prbs7.shape)
# print("prbs7 labels shape:", labels_rnn_prbs7.shape)


# windows_rnn_random, labels_rnn_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# print("random Windows shape", windows_rnn_random.shape)
# print("random labels shape:", labels_rnn_random.shape)

# all_windows_rnn = np.concatenate([windows_rnn_square, windows_rnn_prbs4, windows_rnn_prbs7, windows_rnn_random])
# all_labels_rnn = np.concatenate([labels_rnn_square, labels_rnn_prbs4, labels_rnn_prbs7, labels_rnn_random])
# print(len(all_windows_rnn), len(all_labels_rnn))

print("Windows rnn shape", all_windows_rnn.shape)
print("labels shape:", all_labels_rnn.shape)



train_windows_rnn, test_windows_rnn, train_labels , test_labels = make_train_test_splits(all_windows_rnn, all_labels_rnn, test_split=0.025)
len(train_windows_rnn), len(test_windows_rnn), len(train_labels), len(test_labels)

train_features_dataset_rnn = tf.data.Dataset.from_tensor_slices(train_windows_rnn)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)


test_features_dataset_rnn = tf.data.Dataset.from_tensor_slices(test_windows_rnn)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)


train_dataset_rnn=tf.data.Dataset.zip((train_features_dataset_rnn, train_labels_dataset))
test_dataset_rnn=tf.data.Dataset.zip((test_features_dataset_rnn, test_labels_dataset))

train_dataset_rnn=train_dataset_rnn.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset_rnn=test_dataset_rnn.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


inputs = tf.keras.layers.Input(shape=(train_windows_rnn[0].shape))
x = tf.keras.layers.SimpleRNN(32, activation="relu", return_sequences=True)(inputs) #default tanh
x = tf.keras.layers.GlobalAveragePooling1D()(x)
# x = tf.keras.layers.Dense(128, activation="relu")(x)
# x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = Dense(1)(x)
model_4 = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="SimpleRNN_model")


model_4.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))


history_rnn = model_4.fit(train_dataset_rnn,
                epochs=120,
                validation_data=test_dataset_rnn,
                verbose=0,
                callbacks=[early_stopping, verbose_callback])


model_4.summary()
loss_rnn = history_rnn.history["val_loss"]
plt.figure()
plt.plot(loss_rnn, label="simplernnmodelvalloss")
plt.legend()
plt.show()


# Example call
prediction_horizon = len(test_labels[:300]) 
rnn_predictions, rnn_results = predict_2d(
    model=model_4,
    test_windows_2d=test_windows_rnn,
    test_labels=test_labels,
    test_u=train_df_normalized['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
    window_size=WINDOW_SIZE,
    prediction_horizon=prediction_horizon,
    start_index=0
)


#Validation_data_prediction
prediction_horizon = 400
rnn_predictions, rnn_results = predict_2d(
    model=model_4,
    test_windows_2d=validation_data_windows,
    test_labels=validation_data_labels,
    test_u=validation_df_normalized['base_flow'].values[WINDOW_SIZE:],
    window_size=WINDOW_SIZE,
    prediction_horizon=prediction_horizon,
    start_index = 100
)





#GRU windowed data 


# WINDOW_SIZE=30
# BATCH_SIZE=4
# HORIZON=1

# windows_gru_square, labels_gru_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# print("gru Windows shape", windows_gru_square.shape)
# print("gru labels shape:", labels_gru_square.shape)


# windows_gru_prbs4, labels_gru_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# print("prbs4 Windows shape", windows_gru_prbs4.shape)
# print("prbs4 labels shape:", labels_gru_prbs4.shape)


# windows_gru_prbs7, labels_gru_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# print("prbs7 Windows shape", windows_gru_prbs7.shape)
# print("prbs7 labels shape:", labels_gru_prbs7.shape)


# windows_gru_random, labels_gru_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# print("random Windows shape", windows_gru_random.shape)
# print("random labels shape:", labels_gru_random.shape)

# all_windows_gru = np.concatenate([windows_gru_square, windows_gru_prbs4, windows_gru_prbs7, windows_gru_random])
# all_labels_gru = np.concatenate([labels_gru_square, labels_gru_prbs4, labels_gru_prbs7, labels_gru_random])
# print(len(all_windows_gru), len(all_labels_gru))

# print("Windows grushape", all_windows_gru.shape)
# print("labels shape:", all_labels_gru.shape)



# train_windows_gru, test_windows_gru, train_labels , test_labels = make_train_test_splits(all_windows_gru, all_labels_gru, test_split=0.13)
# len(train_windows_gru), len(test_windows_gru), len(train_labels), len(test_labels)

# train_features_dataset_gru = tf.data.Dataset.from_tensor_slices(train_windows_gru)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)


# test_features_dataset_gru = tf.data.Dataset.from_tensor_slices(test_windows_gru)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)


# train_dataset_gru=tf.data.Dataset.zip((train_features_dataset_gru, train_labels_dataset))
# test_dataset_gru=tf.data.Dataset.zip((test_features_dataset_gru, test_labels_dataset))

# train_dataset_gru=train_dataset_gru.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_gru=test_dataset_gru.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# inputs = tf.keras.layers.Input(shape=(train_windows_gru[0].shape))
# x = tf.keras.layers.GRU(32, activation="relu", return_sequences=True)(inputs) #default tanh
# x = tf.keras.layers.GlobalAveragePooling1D()(x)

# # x = tf.keras.layers.Dense(128, activation="relu")(x)
# # x = tf.keras.layers.Dense(128, activation="relu")(x)

# outputs = Dense(1)(x)
# model_5 = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="GRU_model")

# model_5.compile(loss="mae",
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

# history_gru = model_5.fit(train_dataset_gru,
#                 epochs=50, #25-50 arası
#                 validation_data=test_dataset_gru,
#                 callbacks=[early_stopping])


# model_5.summary()
# loss_gru = history_gru.history["val_loss"]
# plt.figure()
# plt.plot(loss_gru, label="grumodelvalloss")
# plt.legend()
# plt.show()

# # Example call
# prediction_horizon = len(test_labels[:300]) 
# gru_predictions, gru_results = predict_2d(
#     model=model_5,
#     test_windows_2d=test_windows_gru,
#     test_labels=test_labels,
#     test_u=random_normalized_df['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon
# )

# print("Evaluation Metrics:", gru_results)


###LSTM windowed data 


# WINDOW_SIZE=30
# BATCH_SIZE=64
# HORIZON=1

# windows_lstm_square, labels_lstm_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# print("lstm Windows shape", windows_lstm_square.shape)
# print("lstm labels shape:", labels_lstm_square.shape)


# windows_lstm_prbs4, labels_lstm_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# print("prbs4 Windows shape", windows_lstm_prbs4.shape)
# print("prbs4 labels shape:", labels_lstm_prbs4.shape)


# windows_lstm_prbs7, labels_lstm_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# print("prbs7 Windows shape", windows_lstm_prbs7.shape)
# print("prbs7 labels shape:", labels_lstm_prbs7.shape)


# windows_lstm_random, labels_lstm_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# print("random Windows shape", windows_lstm_random.shape)
# print("random labels shape:", labels_lstm_random.shape)

# all_windows_lstm = np.concatenate([windows_lstm_square, windows_lstm_prbs4, windows_lstm_prbs7, windows_lstm_random])
# all_labels_lstm = np.concatenate([labels_lstm_square, labels_lstm_prbs4, labels_lstm_prbs7, labels_lstm_random])
# print(len(all_windows_lstm), len(all_labels_lstm))

# print("Windows grushape", all_windows_lstm.shape)
# print("labels shape:", all_labels_lstm.shape)



# train_windows_lstm, test_windows_lstm, train_labels , test_labels = make_train_test_splits(all_windows_lstm, all_labels_lstm, test_split=0.13)
# len(train_windows_lstm), len(test_windows_lstm), len(train_labels), len(test_labels)

# train_features_dataset_lstm = tf.data.Dataset.from_tensor_slices(train_windows_lstm)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)


# test_features_dataset_lstm = tf.data.Dataset.from_tensor_slices(test_windows_lstm)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)


# train_dataset_lstm=tf.data.Dataset.zip((train_features_dataset_lstm, train_labels_dataset))
# test_dataset_lstm=tf.data.Dataset.zip((test_features_dataset_lstm, test_labels_dataset))

# train_dataset_lstm=train_dataset_lstm.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_lstm = test_dataset_lstm.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# inputs = tf.keras.layers.Input(shape=(train_windows_lstm[0].shape))
# #x = tf.keras.layers.LSTM(32, activation="relu", return_sequences=True)(inputs) #default tanh
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation="relu", return_sequences=True))(inputs) #shape is 64 x 2 because goes both 2 ways 


# x = tf.keras.layers.GlobalMaxPooling1D()(x)
# # x = tf.keras.layers.Dense(128, activation="relu")(x)
# # x = tf.keras.layers.Dense(128, activation="relu")(x)

# outputs = Dense(1)(x)
# model_6 = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="LSTM_model")

# model_6.compile(loss="mae",
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

# history_lstm = model_6.fit(train_dataset_lstm,
#                 epochs=25,
#                 validation_data=test_dataset_lstm,
#                 callbacks=[early_stopping])


# model_6.summary()
# loss_lstm = history_lstm.history["val_loss"]
# plt.figure()
# plt.plot(loss_lstm, label="lstm modelvalloss")
# plt.legend()
# plt.show()

# # Example call
# prediction_horizon = len(test_labels[:300])
# lstm_predictions, lstm_results = predict_2d(
#     model=model_6,
#     test_windows_2d=test_windows_lstm,
#     test_labels=test_labels,
#     test_u=random_normalized_df['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon
# )

# print("Evaluation Metrics:", lstm_results)





# ##!!!N-BEATS MODEL Data Prep model_7 1d trained model!!!
# #make windows of data 
# WINDOW_SIZE=30
# BATCH_SIZE=4

# windows_nbeats_square, labels_nbeats_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# print("nbeats Windows shape", windows_nbeats_square.shape)
# print("nbeats labels shape:", labels_nbeats_square.shape)


# windows_nbeats_prbs4, labels_nbeats_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# print("prbs4 Windows shape", windows_nbeats_prbs4.shape)
# print("prbs4 labels shape:", labels_nbeats_prbs4.shape)


# windows_nbeats_prbs7, labels_nbeats_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# print("prbs7 Windows shape", windows_nbeats_prbs7.shape)
# print("prbs7 labels shape:", labels_nbeats_prbs7.shape)


# windows_nbeats_random, labels_nbeats_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# print("random Windows shape", windows_nbeats_random.shape)
# print("random labels shape:", labels_nbeats_random.shape)

# all_windows_nbeats = np.concatenate([windows_nbeats_square, windows_nbeats_prbs4, windows_nbeats_prbs7, windows_nbeats_random])
# all_labels_nbeats = np.concatenate([labels_nbeats_square, labels_nbeats_prbs4, labels_nbeats_prbs7, labels_nbeats_random])

# # all_windows = np.concatenate([windows_prbs7, windows_random])
# # all_labels = np.concatenate([labels_prbs7, labels_random])
# print(len(all_windows_nbeats), len(all_labels_nbeats))


# #Make data 1D for ARX Model
# windows_nbeats = []
# for window in all_windows_nbeats:
#     window_nbeats = window.T.reshape(WINDOW_SIZE*2,) #window_size*2
#     windows_nbeats.append(window_nbeats)

# windows_nbeats = np.array(windows_nbeats)
# print("Windows nbeats shape", all_windows_nbeats.shape)
# print("labels shape:", all_labels_nbeats.shape)



# #THIS IS FOR ARX Model 1D data #be aware of that: test labels and test u must have to come from same dataset fot this example its random u values dataset
# #length of test labels cannot exceed last dataset length
# train_windows_nbeats, test_windows_nbeats, train_labels , test_labels = make_train_test_splits(windows_nbeats, all_labels_nbeats, test_split=0.13)
# print(len(train_windows_nbeats), len(test_windows_nbeats), len(train_labels), len(test_labels))

# train_features_dataset_nbeats = tf.data.Dataset.from_tensor_slices(train_windows_nbeats)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)

# test_features_dataset_nbeats = tf.data.Dataset.from_tensor_slices(test_windows_nbeats)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)

# train_dataset_nbeats=tf.data.Dataset.zip((train_features_dataset_nbeats, train_labels_dataset))
# test_dataset_nbeats=tf.data.Dataset.zip((test_features_dataset_nbeats, test_labels_dataset))
# #batch and prefetch
# train_dataset_nbeats=train_dataset_nbeats.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_nbeats=test_dataset_nbeats.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# HORIZON=1
# #setting up NBEATS model hyperparams using paper
# N_NEURONS = 64
# N_LAYERS = 3
# N_STACKS = 10

# INPUT_SIZE = WINDOW_SIZE*2  #WINDOW_SIZE(60,) * HORIZON(1,) 2 : FEATUREs
# THETA_SIZE = INPUT_SIZE + HORIZON

# INPUT_SIZE, THETA_SIZE


# #setup instance of n beats 
# nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE, 
#                                  theta_size=THETA_SIZE,
#                                  horizon=HORIZON, 
#                                  n_neurons=N_NEURONS, 
#                                  n_layers=N_LAYERS,
#                                  name="InitialBlock")

# #2. create input to stack
# stack_input = tf.keras.layers.Input(shape=(INPUT_SIZE), name="stack_input")

# #3create initial backcast and forecast 
# backcast, forecast = nbeats_block_layer(stack_input)
# residuals=tf.keras.layers.subtract([stack_input, backcast], name="subtract_00")


# for i, _ in enumerate(range(N_STACKS-1)): #first stack already created in 3
#     #5 use the nbeatsblock to calculate backcast and forecast
#     backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
#                                            theta_size=THETA_SIZE,
#                                            horizon=HORIZON,
#                                            n_neurons=N_NEURONS,
#                                            n_layers=N_LAYERS,
#                                            name=f"NBeatsblock_{i}")(residuals)
#     #6create double residual stacking
#     residuals=tf.keras.layers.subtract([residuals,backcast], name=f"subtract_{i}")
#     forecast=tf.keras.layers.add([forecast, block_forecast], name=f"add_{i}")
    
# #7 put the stack model together
# model_7 = tf.keras.Model(inputs=stack_input, outputs=forecast, name="model_7_NBEATS")

# #8 compile
# model_7.compile(loss="mae",
#                 optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001))

# history_nbeats = model_7.fit(train_dataset_nbeats,
#                 epochs=500,
#                 validation_data=test_dataset_nbeats,
#                 callbacks=[early_stopping])
# save_dir_n = "nbeats_model_experiments"
# os.makedirs(save_dir_n, exist_ok=True)  # Ensure the directory exists
# model_7.save(os.path.join(save_dir_n, "nbeats_model.keras"))


# model_7.summary()
# loss_nbeat = history_nbeats.history["val_loss"]
# plt.figure()
# plt.plot(loss_nbeat, label="nbeatsmodelvalloss")
# plt.legend()
# plt.show()

# # Define custom objects dictionary
# custom_objects = {"NBeatsBlock": NBeatsBlock}
# # Load model with custom objects
# loaded_model = tf.keras.models.load_model(
#     "nbeats_model_experiments/nbeats_model.keras",
#     custom_objects=custom_objects
# )

# prediction_horizon = len(test_labels[:300]) 
# nbeats_predictions, nbeats_results = predict_1d(
#     model=model_7,
#     test_windows_1d=test_windows_nbeats,
#     test_labels=test_labels,
#     test_u=random_normalized_df['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
# )

# print("Evaluation Metrics:", nbeats_results)



# ##ENSEMBLE MODEL Data Prep model_8 1d trained model!!!
# #make windows of data 
# WINDOW_SIZE=30
# BATCH_SIZE=4
# HORIZON = 1

# windows_ens_square, labels_ens_square = make_multivariate_windows(data=square_normalized_df, window_size=WINDOW_SIZE)
# print("ens Windows shape", windows_ens_square.shape)
# print("ens labels shape:", labels_ens_square.shape)


# windows_ens_prbs4, labels_ens_prbs4 = make_multivariate_windows(data=prbs4_normalized_df, window_size=WINDOW_SIZE)
# print("prbs4 Windows shape", windows_ens_prbs4.shape)
# print("prbs4 labels shape:", labels_ens_prbs4.shape)


# windows_ens_prbs7, labels_ens_prbs7 = make_multivariate_windows(data=prbs7_normalized_df, window_size=WINDOW_SIZE)
# print("prbs7 Windows shape", windows_ens_prbs7.shape)
# print("prbs7 labels shape:", labels_ens_prbs7.shape)


# windows_ens_random, labels_ens_random = make_multivariate_windows(data=random_normalized_df, window_size=WINDOW_SIZE)
# print("random Windows shape", windows_ens_random.shape)
# print("random labels shape:", labels_ens_random.shape)

# all_windows_ens = np.concatenate([windows_ens_square, windows_ens_prbs4, windows_ens_prbs7, windows_ens_random])
# all_labels_ens = np.concatenate([labels_ens_square, labels_ens_prbs4, labels_ens_prbs7, labels_ens_random])

# # all_windows = np.concatenate([windows_prbs7, windows_random])
# # all_labels = np.concatenate([labels_prbs7, labels_random])
# print(len(all_windows_ens), len(all_labels_ens))


# #Make data 1D for ensemble Model
# windows_ens = []
# for window in all_windows_ens:
#     window_ens = window.T.reshape(WINDOW_SIZE*2,) #window_size*2
#     windows_ens.append(window_ens)

# windows_ens = np.array(windows_ens)
# print("Windows ens shape", all_windows_ens.shape)
# print("labels shape:", all_labels_ens.shape)



# #THIS IS FOR ensemble Model 1D data #be aware of that: test labels and test u must have to come from same dataset fot this example its random u values dataset
# #length of test labels cannot exceed last dataset length
# train_windows_ens, test_windows_ens, train_labels , test_labels = make_train_test_splits(windows_ens, all_labels_ens, test_split=0.13)
# print(len(train_windows_ens), len(test_windows_ens), len(train_labels), len(test_labels))

# train_features_dataset_ens = tf.data.Dataset.from_tensor_slices(train_windows_ens)
# train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)

# test_features_dataset_ens = tf.data.Dataset.from_tensor_slices(test_windows_ens)
# test_labels_dataset = tf.data.Dataset.from_tensor_slices(test_labels)

# train_dataset_ens=tf.data.Dataset.zip((train_features_dataset_ens, train_labels_dataset))
# test_dataset_ens=tf.data.Dataset.zip((test_features_dataset_ens, test_labels_dataset))
# #batch and prefetch
# train_dataset_ens=train_dataset_ens.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset_ens=test_dataset_ens.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# from tensorflow.keras import layers
# save_dir = "ensemble_model_experiments"
num_iter = 5
loss_fn=["mae", "mse", "mape"]
# #ENSEMBLE MODEL
# def get_ensemble_models(horizon=HORIZON,
#                         train_data=train_dataset_ens,
#                         test_data=test_dataset_ens,
#                         num_iter=num_iter,
#                         num_epochs=250,
#                         loss_fn=loss_fn):
    
#     #turns a list of num_iter models each trained on MAE, MSE, MAPE loss
#     #for example if num_iter = 10 and len(loss_fn)=3, them 30 trained models will be returned..
#     #make empty list
#     ensemble_models = []
#     for i in range(num_iter):
#         #build and fit new model with different loss fn
#         for loss_function in loss_fn:
#             print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model_number{i}")
#             #construct simple model
#             model = tf.keras.Sequential([
#                 layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
#                 layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
#                 layers.Dense(HORIZON)
#                 ])
            
#             #compile with current fn
#             model.compile(loss=loss_function,
#                           optimizer=tf.keras.optimizers.Adam(),
#                           metrics=["mae", "mse"])
#             #fit
#             model.fit(train_data,
#                       epochs=num_epochs,
#                       verbose=1,
#                       validation_data=test_data,
#                       callbacks=[early_stopping,                                 
#                                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
#                                                                      patience=100,
#                                                                      verbose=1)]
#                       )
#             #append fitted model
#             model.save(f"{save_dir}/model_iter{i}_{loss_function}.keras")
#             ensemble_models.append(model)
    
#     return ensemble_models



# model_8 = get_ensemble_models()
# # Load all models into a list
# # Load models by reconstructing filenames
# loss_fns=["mae", "mse", "mape"]
# loaded_ensemble = []
# for iteration in range(num_iter):
#     for loss_function in loss_fn:
#         model_path = f"ensemble_model_experiments/model_iter{iteration}_{loss_function}.keras"
#         model = tf.keras.models.load_model(model_path)
#         loaded_ensemble.append(model)

# prediction_horizon = 50
# ens_predictions, ens_results = predict_ensemble(
#     ensemble_models=model_8,
#     test_windows_1d=test_windows_ens,
#     test_labels=test_labels,
#     test_u=random_normalized_df['base_flow'].values[-len(test_labels):], #this will be correct until len(random_df) if len(test_labels> random df makint test_u like that will be wrong)
#     window_size=WINDOW_SIZE,
#     prediction_horizon=prediction_horizon,
#     start_index=150
# )

# print("Evaluation Metrics:", ens_results)