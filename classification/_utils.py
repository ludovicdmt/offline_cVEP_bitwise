import numpy as np
from scipy.stats import pearsonr

def make_preds_pvalue(y_pred, codes, min_len=50, sfreq=500):
    length = int((2.2 - 0.250) * sfreq)
    y_pred = np.array(y_pred)

    code_buffer = []
    labels_pred = []
    code_pos = 0
    y_tmp = []
    mean_long = []

    for trial in range(int(len(y_pred) / length)):
        # Retrieve a trial
        tmp_code = y_pred[trial * length : (trial + 1) * length]
        code_pos = 0

        # Do an average over the predictions and store it in the code buffer
        # Predictions are averaged because samplig rate is 500Hz and refresh rate 60Hz
        code_buffer = []
        for idx in range(len(tmp_code)):
            y_tmp.append(tmp_code[idx])
            if idx / sfreq >= (code_pos + 1) / 60:
                code_pred = np.mean(y_tmp)
                code_pred = int(np.rint(code_pred))
                code_buffer.append(code_pred)
                y_tmp = []
                code_pos += 1
        # Find the code that correlate the most
        corr = -2
        pred_lab = -1
        for long in np.arange(min_len, len(code_buffer), step=3):
            corrs = []
            pred_lab = -1
            corr = -2
            tmp = -1
            for key, values in codes.items():
                corp, p_value = pearsonr(code_buffer[:long], values[:long])
                corrs.append(corp)
                if corp > corr:
                    corr = corp
                    p_max = p_value
                    tmp = key
            corrs_idx = np.argsort(corrs)
            # If p_value bigger than a treshold and 50% diff between first and second guess
            # Throw a prediciton
            if (
                (corrs[corrs_idx[-1]] - corrs[corrs_idx[-2]]) / corrs[corrs_idx[-1]]
                > 0.5
            ) and (p_max < 1e-5):
                mean_long.append((long) / 60)
                pred_lab = tmp
                break
        labels_pred.append(pred_lab)
    labels_pred = np.array(labels_pred)
    return labels_pred, code_buffer, mean_long


def make_preds_accumul_aggresive(y_pred, codes, min_len=30, sfreq=500):
    length = int((2.2-0.250)*sfreq)
    y_pred = np.array(y_pred)
    rez_acc = []

    code_buffer = []
    labels_pred = []
    code_pos = 0
    y_tmp = [] 
    mean_long = []


    for trial in range(int(len(y_pred)/length)):   
        # Retrieve a trial
        tmp_code = y_pred[trial*length:(trial+1)*length]
        code_pos = 0

        # Do an average over the prdata, codes, labels, sfreq
        code_buffer = []
        for idx in range(len(tmp_code)):
            y_tmp.append(tmp_code[idx])
            if idx/sfreq >= (code_pos+1)/60:
                code_pred = np.mean(y_tmp) 
                code_pred = int(np.rint(code_pred))
                code_buffer.append(code_pred) 
                y_tmp = []
                code_pos += 1
        # Find the code that correlate the most
        corr = -2
        pred_lab = -1
        out = 0
        for long in np.arange(min_len, len(code_buffer) -1 , step=1):
            dtw_values = []
            for key, values in codes.items():
                dtw_values.append(np.corrcoef(code_buffer[:long], values[:long])[0,1])
            dtw_values = np.array(dtw_values)
            max_dtw = list(codes.keys())[np.argmax(dtw_values)] 
            if (max_dtw == pred_lab):
                out += 1
                corr = np.max(dtw_values)
            else:
                pred_lab = max_dtw
                out = 0
            if out == 30:
                mean_long.append((long)/60)
                break
        labels_pred.append(pred_lab)
    labels_pred = np.array(labels_pred)
    return labels_pred, code_buffer, mean_long


def train_split(data, labels, n_cal):
    data_train = data[:11*n_cal]
    labels_train = labels[:11*n_cal]
    data_test = data[11*n_cal:]
    labels_test = labels[11*n_cal:]
    
    return data_train, labels_train, data_test, labels_test


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return np.squeeze(pos_encoding)
