import os
import numpy as np


def batch_iterator(batch_size, data_path, shape):
    """Batch iterator for training CNN"""
    noisy_dir = data_path + '/noisy'
    clean_dir = data_path + '/clean'
    noisy_lst = os.listdir(noisy_dir)
    clean_lst = os.listdir(clean_dir)
    size = max(len(noisy_lst), len(clean_lst))
    data, labels = [], []
    for i in range(len(noisy_lst[:size])):
        path = noisy_dir + '/' + noisy_lst[i]
        files = os.listdir(path)
        for j in range(len(files)):
            data.append(path + '/' + files[j])
            labels.append(1)
    for i in range(len(clean_lst[:size])):
        path = clean_dir + '/' + clean_lst[i]
        files = os.listdir(path)
        for j in range(len(files)):
            data.append(path + '/' + files[j])
            labels.append(0)
    data, labels = np.array(data), np.array(labels)
    idx = np.array(range(len(data)))
    np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        x = np.stack([reshape_mel(np.load(data[idx[j]]), shape).astype('float32')[None] for j in
                      range(i, min(i + batch_size, len(data)))], 0)
        y = np.stack([labels[idx[j]] for j in range(i, min(i + batch_size, len(data)))], 0)
        yield x, y


def reshape_mel(mel, shape=(80, 80)):
    """Reshape MEL-spectogram Using Simple Method Pad/Trim"""
    if mel.shape[0] > shape[0]:
        diff = mel.shape[0] - shape[0]
        offset = np.random.randint(diff)
        mel = mel[offset:shape[0] + offset, :]
    elif mel.shape[0] < shape[0]:
        diff = shape[0] - mel.shape[0]
        offset = np.random.randint(diff)
        mel = np.pad(mel, ((offset, shape[0] - mel.shape[0] - offset), (0, 0)), "constant")
    if mel.shape[1] > shape[1]:
        diff = mel.shape[1] - shape[1]
        offset = np.random.randint(diff)
        mel = mel[:, offset:shape[1] + offset]
    elif mel.shape[1] < shape[1]:
        diff = shape[1] - mel.shape[1]
        offset = np.random.randint(diff)
        mel = np.pad(mel, ((0, 0), (offset, shape[1] - mel.shape[1] - offset)), "constant")
    return mel
