from EEG2CodeKeras import (basearchi, 
basearchi_patchembedding, 
basearchi_patchembeddingdilation, 
vanilliaEEG2Code, 
vanilliaEEG2Code2)
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from load_data import load_data_neg

from _utils import make_preds_accumul_aggresive, make_preds_pvalue

if __name__ == "__main__":
    path = "/path/to/data/set/"
    n_cal = 7
    models = basearchi, basearchi_patchembedding, basearchi_patchembeddingdilation, \
        vanilliaEEG2Code, vanilliaEEG2Code2
    lr = 1e-3
    batchsize = 256

    weight_decay = 1e-4
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

        if (model.__name__ == "basearchi_patchembedding") or (model.__name__ == "basearchi_patchembeddingdilation") or (model.__name__ == "basearchi_patchembeddingdepthwise"):
            epochs = 35
        elif (model.__name__ == "convMixer"):
            epochs = 15
        else:
            epochs = 45

        for sub in range(11):
            print(f"Subject number {sub}")
            #  load data
            X_train, y_train, X_test, y_test, labels_test, X_neg, labels_neg, \
                codes, n_trial_per_class, sfreq = load_data_neg(sub=sub, path=path, n_cal=n_cal)
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
            optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)

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

            pred = clf.predict(X_test, batch_size=512)
            y_pred = np.array(pred[:, 0])

            pred_neg = clf.predict(X_neg, batch_size=512)
            y_pred_neg = np.array(pred_neg[:, 0])

            # Convert these windows-based prediction to sample predictions
            labels_pred, _, mean_long = make_preds_pvalue(
                y_pred, codes, min_len=70, sfreq=sfreq
            )

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred, codes, min_len=30, sfreq=sfreq 
            )

            labels_pred_neg, _, _ = make_preds_pvalue(
                y_pred_neg, codes, min_len=70, sfreq=sfreq
            )

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

        # Store perf results
        model_std_accumul = np.std(model_acc*100)
        model_acc = np.round(np.mean(model_acc) * 100, 2)
        model_acc_accumul = np.round(np.mean(model_acc_accumul) * 100, 2)
        model_acc_neg = np.round(np.mean(model_acc_neg) * 100, 2)
        model_meanlong = np.round(np.mean(model_meanlong), 2)
        
        print(model_std_accumul)
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
        df.to_csv(f"./results/archi_comparison_with_{n_cal}_cal_block.csv")
