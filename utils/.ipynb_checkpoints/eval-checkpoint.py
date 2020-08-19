from __future__ import absolute_import, print_function
import os
import scipy.io as sio
import numpy as np
import sklearn.metrics as skmetr
import utils
import matplotlib.pyplot as plt


def eval_video2(gt_file, score_path, data):
    label_orig = np.load(gt_file, allow_pickle=True)
    if type(score_path) is str:
        score_orig = np.load(score_path)
    else:
        score_orig = score_path
    score_after, label_after = [], []
    init = 0
    for i in range(len(label_orig)):
        _label_use = label_orig[i]
        _label_use = _label_use[8:-7]
        _score_use = score_orig[init:init+len(_label_use)]
        init += len(_label_use)
        _score_use = _score_use - np.min(_score_use)
        _score_use = 1 - _score_use / np.max(_score_use)
        score_after.append(_score_use)
        label_after.append(1 - _label_use + 1)
    score_after = np.concatenate(score_after, axis=0)
    label_after = np.concatenate(label_after, axis=0)
    print("Number of gt frames:", len(label_after))
    print("Number of predictions:", len(score_after))
    fpr, tpr, thresholds = skmetr.roc_curve(label_after, score_after, pos_label=2)
    auc = skmetr.auc(fpr, tpr)
    print("AUC score on data %s is %.2f" % (data, auc))



def eval_video(data_path, res_path, is_show=False):
    gt_path = os.path.join(data_path, 'testing_gt/')

    ###
    video_list = utils.get_file_list(gt_path, is_sort=True)
    video_num = len(video_list)

    gt_labels_list = []
    res_prob_list = []
    res_prob_list_org = []

    ###
    for vid_ite in range(video_num):
        gt_file_name = video_list[vid_ite]

        p_idx = [pos for pos, char in enumerate(gt_file_name) if char == '.']
        video_name = gt_file_name[0:p_idx[0]]
        print('Eval: %d/%d-%s' % (vid_ite + 1, video_num, video_name))
        # res file name
        res_file_name = video_name + '.npy'
        # gt file and res file - path
        gt_file_path = os.path.join(gt_path, gt_file_name)
        res_file_path = os.path.join(res_path, res_file_name)
        #     print(gt_file_path)
        #     print(res_file_path)

        # read data
        gt_labels = sio.loadmat(gt_file_path)['l'][0]  # ground truth labels
        res_prob = np.load(res_file_path)  # estimated probability scores
        #     res_prob = np.log10(res_prob)-2*np.log10(255)

        res_prob_list_org = res_prob_list_org + list(res_prob)
        gt_labels_res = gt_labels[8:-7]

        # normalize regularity score
        res_prob_norm = res_prob - res_prob.min()
        res_prob_norm = 1 - res_prob_norm / res_prob_norm.max()

        ##
        gt_labels_list = gt_labels_list + list(1 - gt_labels_res + 1)
        res_prob_list = res_prob_list + list(res_prob_norm)

    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=2)
    auc = skmetr.auc(fpr, tpr)
    print(('auc:%f' % auc))

    # output_path = os.path.join(res_path,)
    output_path = res_path
    sio.savemat(os.path.join(output_path, video_name + '_gt_label.mat'),  {'gt_labels_list': np.double(gt_labels_res)}  )
    sio.savemat(os.path.join(output_path, video_name + '_est_label.mat'), {'est_labels_list': np.double(res_prob_list)} )
    acc_file = open(os.path.join(output_path, 'acc.txt'), 'w')
    acc_file.write( '{}\nAUC: {}\n'
              .format(data_path, auc ))
    acc_file.close()

    if(is_show):
        plt.figure()
        plt.plot(gt_labels_list)
        plt.plot(res_prob_list)

    return auc