import mne
import numpy as np
import os
from collections import OrderedDict
from imblearn.under_sampling import RandomUnderSampler
from _utils import positional_encoding


def load_data(sub, path, n_cal=7, reverse=False):

    # Load epoched data
    data, codes, labels, sfreq = load_epoched_data(sub=sub, path=path)
    n_trial_per_class = int(len(data) / 11)

    # Calibration and train split
    if reverse:
        data_train = data[-11 * n_cal:]
        labels_train = labels[-11 * n_cal:]
        data_test = data[:-11 * n_cal]
        labels_test = labels[:-11 * n_cal]
    else:
        data_train = data[:11 * n_cal]
        labels_train = labels[:11 * n_cal]
        data_test = data[11 * n_cal:]
        labels_test = labels[11 * n_cal:]

    # Extract windows of 250ms from trials
    X_train, y_train = to_window(data_train, labels_train, sfreq, codes)
    X_test, y_test = to_window(data_test, labels_test, sfreq, codes)

    # Normalization
    X_std = X_train.std(axis=0)
    X_train /= X_std + 1e-8
    X_test /= X_std + 1e-8

    # Balance the classes for the training
    rus = RandomUnderSampler()
    counter = np.array(range(0, len(y_train))).reshape(-1, 1)
    index, _ = rus.fit_resample(counter, y_train[:, 0])
    X_train = np.squeeze(X_train[index, :, :, :], axis=1)
    y_train = np.squeeze(y_train[index])

    # Mixup 

    # perc_mixup = 0.1
    # alpha = 0.2

    # idx_train = np.arange(len(X_train))
    # to_add = int(len(X_train) * perc_mixup)

    # X_mix = np.zeros((to_add, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    # y_mix = np.zeros((to_add, y_train.shape[1]))

    # for idx_mix in range(to_add):
    #     lambda_rdm = np.random.beta(alpha, alpha)
    #     idx_pick = np.random.choice(idx_train, size = 2, replace = False)
    #     X_mix[idx_mix] = lambda_rdm * X_train[idx_pick[0]] + (1 - lambda_rdm) * X_train[idx_pick[1]]
    #     y_mix[idx_mix] = lambda_rdm * y_train[idx_pick[0]] + (1 - lambda_rdm) * y_train[idx_pick[1]]

    # X_train = np.concatenate((X_train, X_mix), axis=0)
    # y_train = np.concatenate((y_train, y_mix), axis=0)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        labels_test,
        codes,
        n_trial_per_class,
        sfreq,
    )


def load_epoched_data(sub, path):
    """Load the .set and convert it to epoched data"""
    subjects = ["P"+str(i) for i in range(11)]
    sub = subjects[sub]
    file_name = f"{sub}_mseqwhite.set"

    #  load the data with MNE
    raw = mne.io.read_raw_eeglab(
        os.path.join(path, file_name), preload=True, verbose=False
    )
    ch_without_ACC = list(map(str, range(1, 33)))  # Drop the accelero channels
    raw = raw.drop_channels(list(set(raw.ch_names) - set(ch_without_ACC)))
    raw = raw.drop_channels(["21", "10"])  # Channels to noisy close to the ears
    mne.set_eeg_reference(raw, "average", copy=False, verbose=False)
    
    # Line-noise filtering
    raw = raw.filter(l_freq=50.1, h_freq=49.9, method="iir", verbose=False) 

    # Band-pass filtering
    #raw = raw.filter(l_freq=5, h_freq=45, method="fir", verbose=False) 

    # Reformate the annotation to remove some char
    for idx in range(len(raw.annotations.description)):
        code = raw.annotations.description[idx].split("_")[0]
        lab = raw.annotations.description[idx].split("_")[1]
        code = code.replace("\n", "")
        code = code.replace("[", "")
        code = code.replace("]", "")
        code = code.replace(" ", "")
        raw.annotations.description[idx] = code + "_" + lab

    # Epoching of the data
    events, event_id = mne.events_from_annotations(raw, event_id="auto", verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=+2.2,
        baseline=(None, None),
        preload=False,
        verbose=False,
    )
    labels = epochs.events[..., -1]
    labels -= np.min(labels)
    data = epochs.get_data()

    codes = OrderedDict()
    for k, v in event_id.items():
        code = k.split("_")[0]
        idx = k.split("_")[1]
        codes[v - 1] = np.array(list(map(int, code)))

    sfreq = int(epochs.info["sfreq"])

    return data, codes, labels, sfreq

def load_data_neg(sub, path, n_cal=7):

    # Load epoched data
    data, data_neg, codes, labels, labels_neg, sfreq = load_epoched_data_neg(sub=sub, path=path)
    n_trial_per_class = int(len(data) / 11)

    # Calibration and train split
    data_train = data[:11 * n_cal]
    labels_train = labels[:11 * n_cal]
    data_test = data[11 * n_cal:]
    labels_test = labels[11 * n_cal:]

    # Extract windows of 250ms from trials
    X_train, y_train = to_window(data_train, labels_train, sfreq, codes)
    X_test, y_test = to_window(data_test, labels_test, sfreq, codes)
    X_neg =  to_window_neg(data_neg, sfreq)

    # Normalization
    X_std = X_train.std(axis=0)
    X_train /= X_std + 1e-8
    X_test /= X_std + 1e-8
    X_neg /= X_std + 1e-8

    # Balance the classes for the training
    rus = RandomUnderSampler()
    counter = np.array(range(0, len(y_train))).reshape(-1, 1)
    index, _ = rus.fit_resample(counter, y_train[:, 0])
    X_train = np.squeeze(X_train[index, :, :, :], axis=1)
    y_train = np.squeeze(y_train[index])

    # # Mixup 
    # perc_mixup = 0.05
    # alpha = 0.3

    # idx_train = np.arange(len(X_train))
    # to_add = int(len(X_train) * perc_mixup)

    # X_mix = np.zeros((to_add, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    # y_mix = np.zeros((to_add, y_train.shape[1]))

    # for idx_mix in range(to_add):
    #     lambda_rdm = np.random.beta(alpha, alpha)
    #     idx_pick = np.random.choice(idx_train, size = 2, replace = False)
    #     X_mix[idx_mix] = lambda_rdm * X_train[idx_pick[0]] + (1 - lambda_rdm) * X_train[idx_pick[1]]
    #     y_mix[idx_mix] = lambda_rdm * y_train[idx_pick[0]] + (1 - lambda_rdm) * y_train[idx_pick[1]]

    # X_train = np.concatenate((X_train, X_mix), axis=0)
    # y_train = np.concatenate((y_train, y_mix), axis=0)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        labels_test,
        X_neg,
        labels_neg,
        codes,
        n_trial_per_class,
        sfreq,
    )

def load_epoched_data_neg(sub, path):
    """Load the .set, select only the inter-trials and convert it to epoched data"""
    subjects = ["P"+str(i) for i in range(11)]
    sub = subjects[sub]

    file_name = f"{sub}_mseqwhite.set"

    #  load the data with MNE
    raw = mne.io.read_raw_eeglab(
        os.path.join(path, file_name), preload=True, verbose=False
    )
    raw = raw.drop_channels(["21", "10"])
    mne.set_eeg_reference(raw, "average", copy=False, verbose=False)
    ch_without_ACC = list(map(str, range(1, 33)))  # Drop the accelero channels
    raw = raw.drop_channels(list(set(raw.ch_names) - set(ch_without_ACC)))
    raw = raw.filter(l_freq=50.1, h_freq=49.9, method="iir", verbose=False)  # Filtering
    #raw = raw.filter(l_freq=5, h_freq=45, method="fir", verbose=False)  # Filtering

    # Reformate the annotation to remove some char
    for idx in range(len(raw.annotations.description)):
        code = raw.annotations.description[idx].split("_")[0]
        lab = raw.annotations.description[idx].split("_")[1]
        code = code.replace("\n", "")
        code = code.replace("[", "")
        code = code.replace("]", "")
        code = code.replace(" ", "")
        raw.annotations.description[idx] = code + "_" + lab

    # Epoching of the data
    events, event_id = mne.events_from_annotations(raw, event_id="auto", verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=+2.2,
        baseline=(None, None),
        preload=False,
        verbose=False,
    )
    labels = epochs.events[..., -1]
    labels -= np.min(labels)
    data = epochs.get_data()

    epochs_neg = mne.Epochs(raw, events, event_id=event_id, tmin=-2.0, \
            tmax=+0.2, baseline=(None, None), preload=False, verbose=False)
    data_neg = epochs_neg.get_data()
    labels_neg = np.array([-1]*len(data_neg))

    codes = OrderedDict()
    for k, v in event_id.items():
        code = k.split("_")[0]
        idx = k.split("_")[1]
        codes[v - 1] = np.array(list(map(int, code)))

    sfreq = int(epochs.info["sfreq"])

    return data, data_neg, codes, labels, labels_neg, sfreq


def code2array(code):
    """Convert the code from a string to an array"""
    tmp = []
    for idx, c in enumerate(code[:-2]):
        if c == "5" or c == ".":
            continue
        elif c == "0":
            if code[idx + 2] == "5":
                tmp.append(0.5)
            else:
                tmp.append(0)
        else:
            tmp.append(1)
    if code[-1] == ".":
        if code[-2] == "0":
            tmp.append(0)
        else:
            tmp.append(1)
    return np.array(tmp)


def to_window(data, labels, sfreq, codes):
    n_channels = data.shape[1]
    n_samples_windows = int(0.250 * sfreq)
    length = int((2.2 - 0.250) * sfreq)
    X = np.empty(shape=((length) * data.shape[0], n_channels, n_samples_windows))
    y = np.empty(shape=((length) * data.shape[0]), dtype=int)
    count = 0
    #pos_encoding = positional_encoding(int((2.2-0.250)*60), n_samples_windows)

    for trial_nb, trial in enumerate(data):
        lab = labels[trial_nb]
        c = codes[lab]
        code_pos = 0
        for idx in range(length):
            X[count] = trial[:, idx : idx + n_samples_windows]
            #X[count, :-1, :] = trial[:, idx:idx+n_samples_windows]
            #X[count, -1, :] = pos_encoding[code_pos]
            if idx / sfreq >= (code_pos + 1) / 60:
                code_pos += 1
            y[count] = int(c[code_pos])
            count += 1

    X = np.expand_dims(X, 1)
    X = X.astype(np.float32)
    y = np.vstack((y, np.abs(1 - y))).T
    return X, y

def to_window_neg(data, sfreq):
    n_channels = data.shape[1]
    n_samples_windows = int(0.250*sfreq)
    length = int((2.2-0.250)*sfreq)
    X = np.empty(shape=((length)*data.shape[0], n_channels, n_samples_windows))
    count = 0
    # pos_encoding = positional_encoding(int((2.2-0.250)*60), n_samples_windows)
    for _, trial in enumerate(data):
        code_pos = 0
        for idx in range(length):
            X[count] = trial[:, idx:idx+n_samples_windows]
            # X[count, :-1, :] = trial[:, idx:idx+n_samples_windows]
            # X[count, -1, :] = pos_encoding[code_pos]
            if idx/sfreq >= (code_pos+1)/60:
                code_pos += 1
            count += 1

    X = np.expand_dims(X, 1)
    X = X.astype(np.float32)
    return X
