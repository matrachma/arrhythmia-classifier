from signal_processing import bwr
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ELU
from keras import optimizers as opt
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras_adabound import AdaBound
from PIL import Image
import numpy as np
import wfdb
import os
import biosppy
import cv2
import re
import matplotlib.pyplot as plt
import warnings

class_image = ["F", "N", "Q", "SVEB", "VEB"]
label_map = {class_image[i]: i for i in range(len(class_image))}
class_map = {i: class_image[i] for i in range(len(class_image))}


def load_saved_model(optm='', path=''):
    image_height = 128
    image_width = 128
    num_channels = 1
    the_model = Sequential()
    # add Convolution layers
    the_model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                         input_shape=(image_height, image_width, num_channels), kernel_initializer='glorot_uniform'))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    the_model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    the_model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    the_model.add(Flatten())
    the_model.add(Dense(2048))
    the_model.add(ELU())
    the_model.add(BatchNormalization())
    the_model.add(Dropout(0.5))
    the_model.add(Dense(5, activation='softmax'))

    if optm == "sgd":
        the_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    elif optm == "adam":
        the_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif optm == "adagrad":
        the_model.compile(loss='categorical_crossentropy', optimizer=opt.Adagrad(lr=0.01, epsilon=None, decay=0.0),
                          metrics=['accuracy'])
    elif optm == "adabound":
        the_model.compile(loss='categorical_crossentropy', optimizer=AdaBound(lr=1e-03,
                                                                              final_lr=0.1,
                                                                              gamma=1e-03,
                                                                              weight_decay=0.,
                                                                              amsgrad=False), metrics=['accuracy'])
    elif optm == "amsbound":
        the_model.compile(loss='categorical_crossentropy', optimizer=AdaBound(lr=1e-03,
                                                                              final_lr=0.1,
                                                                              gamma=1e-03,
                                                                              weight_decay=0.,
                                                                              amsgrad=True), metrics=['accuracy'])
    elif optm == "adadelta":
        the_model.compile(loss='categorical_crossentropy',
                          optimizer=opt.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                          metrics=['accuracy'])

    if os.path.isfile(path):
        the_model.load_weights(path)
        the_model._make_predict_function()
    else:
        warnings.warn("Warning: File {} is not exist, you will use model with random weights".format(path))

    return the_model


def ecg_to_images(patient):
    symbol_map = {
        "N": ['N', '.', 'L', 'R', 'e', 'j'],
        "SVEB": ['A', 'a', 'J', 'S'],
        "VEB": ['V', 'E'],
        "F": ['F'],
        "Q": ['P', '/', 'f', 'U']
    }
    data_image = "temp/images/"
    record = os.path.join("temp/uploaded/", patient)
    signal0, field0 = wfdb.rdsamp(record, channels=[0])
    baseline0, signal0_bwr = bwr(signal0[:, 0])
    filtered_signal0_bwr, _, _ = biosppy.tools.filter_signal(signal=signal0_bwr, ftype='FIR', band='bandpass',
                                                             order=int(0.3 * 100), frequency=[0.5, 12],
                                                             sampling_rate=100)
    anotasi = wfdb.rdann(record, 'atr')
    anotasi_simbol = list(anotasi.symbol)
    all_selected_point = []
    ids = 0
    skipped_ids = []
    beat_class = ""
    print("Processing patient: ", patient)
    os.makedirs(os.path.join("temp/images/", patient))
    print("Total anotasi: ", str(len(anotasi_simbol)))
    for i, simbol in enumerate(anotasi_simbol):
        # Skip first and last annotation
        if i == 0 or i == len(anotasi_simbol) - 1 or simbol == '+':
            skipped_ids.append((i, simbol))
            continue
        # Get beat class
        for c in list(symbol_map.keys()):
            if simbol in symbol_map[c]:
                beat_class = c

        # Segmentation
        r_peak = anotasi.sample[i]
        r_peak_prv = anotasi.sample[i - 1]  # r peak previous
        r_peak_fwd = anotasi.sample[i + 1]  # r peak next
        start = r_peak - int(abs(r_peak - r_peak_prv) / 2)
        end = r_peak + int(abs(r_peak_fwd - r_peak) / 2)
        selected_point = [i for i in range(start, end)]
        all_selected_point += selected_point
        all_selected_point = list(set(all_selected_point))
        segmented_filtered_signal0_bwr = filtered_signal0_bwr[selected_point]
        # segmented_filtered_signal1_bwr = filtered_signal1_bwr[selected_point]
        ids += 1

        # Create image
        fig = plt.figure(frameon=False, figsize=(1.28, 1.28), dpi=100)
        plt.plot(segmented_filtered_signal0_bwr)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # Save Image
        if simbol == '/':
            simbol = 'slash'
        elif simbol == '.':
            simbol = 'dot'

        filename = str(ids) + '-' + simbol + '-' + beat_class + '.png'
        full_path_file = os.path.join(data_image, patient, filename)
        fig.savefig(full_path_file)
        plt.close(fig)

        # Convert to greyscale
        ii = cv2.imread(full_path_file)
        gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(full_path_file, gray_image)

    print("Done")


def load_data(patient_no):
    image_dir = os.path.join("temp/images/", patient_no)

    ts_data = []
    ts_label = []
    ts_file_path = []
    ts_beat_id = []
    ts_symbol = []

    list_file = os.listdir(image_dir)

    for f in list_file:
        full_path_file = os.path.join(image_dir, f)
        filename_wo_extension = str(f.split(".")[0])
        split_filename = filename_wo_extension.split("-")
        beat_id, beat_symbol, beat_class = split_filename[0], split_filename[1], split_filename[2]
        ts_label.append(label_map[beat_class])
        ts_data.append(np.asarray(Image.open(full_path_file)))
        ts_file_path.append(full_path_file)
        ts_beat_id.append(int(beat_id))
        ts_symbol.append(beat_symbol)

    return np.asarray(ts_data), np.asarray(ts_label), ts_file_path, ts_beat_id, ts_symbol


def evaluate_model(model_dict, patient, model_selected):
    image_dir = os.path.join("temp/images/", patient)
    the_model = model_dict[model_selected]
    if not os.path.exists(image_dir):
        ecg_to_images(patient)
    data = []
    response = None

    if len(os.listdir(image_dir)) > 0:
        image_height = 128
        image_width = 128
        num_channels = 1
        num_classes = 5
        ts_data, ts_label, file_paths, beat_ids, beat_symbols = load_data(patient)

        # one-hot encode the labels - we have 10 output classes
        # so 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0] & so on
        test_labels_cat = to_categorical(ts_label, num_classes)
        del ts_label

        test_data = np.reshape(ts_data, (ts_data.shape[0], image_height, image_width, num_channels))
        del ts_data

        # re-scale the image data to values between (0.0,1.0]
        test_data = test_data.astype('float32') / 255.

        y_pred = the_model.predict(test_data, 1, 1)
        y_pred = y_pred.argmax(axis=1)
        y_true = test_labels_cat.argmax(axis=1)

        all_class_true_this_task = sorted(list(set(y_true)))
        all_class_pred_this_task = sorted(list(set(y_pred)))
        target_names = sorted(list(set(all_class_pred_this_task + all_class_true_this_task)))
        target_names = [class_map[i] for i in target_names]

        miss_classified = 0

        for i, true_label in enumerate(y_true):
            flag = "Correct"
            predicted_class = class_map[y_pred[i]]
            true_class = class_map[true_label]

            if predicted_class != true_class:
                miss_classified += 1
                flag = "Wrong"

            data.append({
                "id_beat": beat_ids[i],
                "beat_symbol": beat_symbols[i],
                "file_image": file_paths[i],
                "predicted_class": predicted_class,
                "true_class": true_class,
                "flag": flag
            })

        data = sorted(data, key=lambda i: i["id_beat"])

        response = {
            "data": data,
            "confusion_matrix": preprocess_report(confusion_matrix(y_true, y_pred)),
            "classification_report": preprocess_report(
                classification_report(y_true, y_pred, target_names=target_names)),
            "total_classified": str(len(data)),
            "miss_classified": str(miss_classified),
        }

    return response


def preprocess_report(report):
    if type(report) != str:
        report = str(report)

    processed = re.sub(r" ", "&nbsp;", report)
    processed = re.sub("\n\n", "\n", processed)
    processed = re.sub("\n", "<br>", processed)

    return processed
