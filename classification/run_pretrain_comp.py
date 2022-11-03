from EEG2CodeKeras import basearchi, basearchi_patchembedding, basearchi_patchembeddingdilation, \
    vanilliaEEG2Code, vanilliaEEG2Code2
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
import shutil
from imblearn.under_sampling import RandomUnderSampler

from load_data import load_epoched_data, to_window, load_data_neg

from _utils import make_preds_accumul_aggresive, make_preds_pvalue, train_split

if __name__ == "__main__":
    path = "/path/to/data/set/"
    n_cal = 5
    lr = 1e-5
    batchsize = 256
    pre_train_list = np.array([4, 1])
    models = basearchi, basearchi_patchembedding, basearchi_patchembeddingdilation, \
        vanilliaEEG2Code, vanilliaEEG2Code2

    results = {}

    for model in models:
        print(f"Running computation for {model.__name__}")
        model_acc = []
        model_acc_accumul = []
        model_acc_neg = []
        model_meanlong = []
        model_noclassif = []
        model_test_decod_acc = []
        model_val_decod_acc = []

        data, codes, labels, sfreq =  load_epoched_data(pre_train_list[0], path)
        n_channels = data.shape[1]
        n_trial_per_class = int(len(data)/11)

        data_train, labels_train, data_test, labels_test = train_split(data, labels, 13)
        X_train, y_train = to_window(data_train, labels_train, sfreq, codes)
        X_test, y_test = to_window(data_test, labels_test, sfreq, codes)

        # Load data from the subjects used for the pre-training
        for sub in pre_train_list[1:]:
            print(f"Subject number {sub}")
            #  load data
            data, codes2, labels, sfreq =  load_epoched_data(pre_train_list[0], path)
            data_train, labels_train, data_test, labels_test = train_split(data, labels, 13)
            X_train2, y_train2 = to_window(data_train, labels_train, sfreq, codes2)
            X_test2, y_test2 = to_window(data_test, labels_test, sfreq, codes2)

            codes = {**codes, **codes2}

            X_train = np.vstack((X_train, X_train2))
            y_train = np.vstack((y_train, y_train2))
            X_test = np.vstack((X_test, X_test2))
            y_test = np.vstack((y_test, y_test2))

        # Normalize data
        X_std = X_train.std(axis=0)
        X_train /= X_std + 1e-8
        X_test /= X_std + 1e-8

        # Shuffle data
        assert len(X_train) == len(y_train)
        p = np.random.permutation(len(X_train))
        X_train=X_train[p]
        y_train=y_train[p]

        # Balance training set
        rus = RandomUnderSampler()
        counter = np.array(range(0, len(y_train))).reshape(-1, 1)
        index, _ = rus.fit_resample(counter, y_train[:, 0])
        X_train = np.squeeze(X_train[index, :, :, :], axis=1)
        y_train = np.squeeze(y_train[index])

        # Define classifier
        n_samples_windows = X_train.shape[-1]
        n_channels = X_train.shape[-2]

        clf = model(windows_size=n_samples_windows, n_channel_input=n_channels)

        # Define validation set and some parameters for the training
        X_train, x_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, shuffle=True
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batchsize)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batchsize)

        if (model.__name__ == "basearchi_patchembedding") or (model.__name__ == "basearchi_patchembeddingdilation") or (model.__name__ == "basearchi_patchembeddingdepthwise"):
            epochs = 4
        else:
            epochs = 5
        # Compile and train the model

        # Freeze some layers
        for lay in clf.layers:
            if 'conv2d_3' in lay.name:
                lay.trainable = False
            elif 'batch_normalization_3' in lay.name:
                lay.trainable = False
            elif 'dense' in lay.name:
                lay.trainable = False

        weight_decay = 1e-4
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        # optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)

        clf.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        history = clf.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            shuffle=True,
            verbose=0,
        )
        decod_val_acc = history.history["val_accuracy"][-1]

        sub_predict = list(set(list(range(10))) - set(pre_train_list))

       # Unfreeze the layers
        for lay in clf.layers:
            if  'conv2d_3' in lay.name:
                lay.trainable = True
            elif 'batch_normalization_3' in lay.name:
                lay.trainable = True
            elif 'dense' in lay.name:
                lay.trainable = True

         # Save the pre-trained model
        model_name = './pretrained' + str(time.time())[-10:].replace('.','') + model.__name__ 
        clf.save(model_name)
        batchsize = 256
        # Train now fully the model for each subject
        for sub in sub_predict:
            print(f"Subject number {sub}")
            #  load data
            X_train, y_train, X_test, y_test, labels_test, X_neg, labels_neg, \
                codes, n_trial_per_class, sfreq = load_data_neg(sub=sub, path=path, n_cal=n_cal)
            # Define classifier
            n_samples_windows = X_train.shape[-1]
            n_channels = X_train.shape[-2]

            # Define validation set and some parameters for the training
            X_train, x_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, shuffle=True
            )

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.batch(batchsize)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(batchsize)

            if (model.__name__ == "basearchi_patchembedding") or (model.__name__ == "basearchi_patchembeddingdilation") or (model.__name__ == "basearchi_patchembeddingdepthwise"):
                epochs = 30
            elif (model.__name__ == "convMixer"):
                epochs = 15
            else:
                epochs = 40

            # Load pre-trained model and train it fully
            clf = keras.models.load_model(model_name)

            lr = 1e-3
            keras.backend.set_value(clf.optimizer.learning_rate, lr)
            history = clf.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                shuffle=True,
                verbose=0,
            )
            decod_val_acc = history.history["val_accuracy"][-1]

            # Convert these windows-based prediction to sample predictions
            pred = clf.predict(X_test, batch_size=512)
            y_pred = np.array(pred[:, 0])

            pred_neg = clf.predict(X_neg, batch_size=512)
            y_pred_neg = np.array(pred_neg[:, 0])

            # Convert these windows-based prediction to sample predictions
            labels_pred, _, mean_long = make_preds_pvalue(y_pred, codes, min_len=100, sfreq=sfreq)

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(y_pred, codes, min_len=30, sfreq=500)

            labels_pred_neg, _, _ = make_preds_pvalue(y_pred_neg, codes, min_len=100, sfreq=sfreq)


            # Compute metrics of performance
            acc = accuracy_score(labels_test, labels_pred)
            acc_accumul = accuracy_score(labels_test, labels_pred_accumul)
            acc_neg = accuracy_score(labels_neg, labels_pred_neg)
            no_classif_p = np.round(100*len(labels_pred[labels_pred==-1])/len(labels_pred),2)
            mean_long = np.mean(mean_long_accumul)
            y_bin = np.copy(y_pred)
            y_bin[y_bin > 0.5] = 1
            y_bin[y_bin <= 0.5] = 0
            decod_acc = accuracy_score(y_test[:, 0], y_bin)

            model_acc.append(acc)
            model_acc_accumul.append(acc_accumul)
            model_acc_neg.append(acc_neg)
            model_meanlong.append(mean_long)
            model_noclassif.append(no_classif_p)
            model_test_decod_acc.append(decod_acc)
            model_val_decod_acc.append(decod_val_acc)

            # Clear variable to free memory
            del clf, history, X_train, y_train, X_test, y_test, y_pred
            keras.backend.clear_session()

        # Clean the directory
        try:
            shutil.rmtree(model_name)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))


        # Store perf results
        model_std_accumul = np.std(model_acc*100)
        model_acc = np.round(np.mean(model_acc) * 100, 2)
        model_acc_accumul = np.round(np.mean(model_acc_accumul) * 100, 2)
        model_acc_neg = np.round(np.mean(model_acc_neg) * 100, 2)
        model_meanlong = np.round(np.mean(model_meanlong), 2)
        model_noclassif = np.round(np.mean(model_noclassif), 2)
        model_test_decod_acc = np.round(np.mean(model_test_decod_acc) * 100, 2)
        model_val_decod_acc = np.round(np.mean(model_val_decod_acc) * 100, 2)

        results[model.__name__] = {
            "overall_acc_pvalue": model_acc,
            "overall_acc_accumul": model_acc_accumul,
            "overall_std_accumul": model_std_accumul,
            "resting_acc_pvalue": model_acc_neg,
            "no_classif_p": model_noclassif,
            "mean_long": model_meanlong,
            "testing_decoding_acc": model_test_decod_acc,
            "validation_decoding_acc": model_val_decod_acc,
        }
        df = pd.DataFrame(results).T
        df.to_csv(f"./results/archi_comparison_pre_training_with_{n_cal}_cal_block.csv")
