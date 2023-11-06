# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle
from sklearn.metrics import roc_curve, auc,confusion_matrix, accuracy_score, recall_score, precision_score
from numpy import interp

#n_classes = 3
y_test = np.array(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
y_score = np.array([[-3.58459897, -0.31176426, 1.78242707],
                    [-2.15411929, 1.11402775, -2.393737],
                    [1.89199335, -3.89624382, -6.29685764],
                    [-4.52609987, -0.63389114, 1.96065819],
                    [1.39684192, -1.77742447, -6.26300472],
                    [-4.29544529, -1.45403694, 3.29458805],
                    [1.60323992, -2.10436714, -6.37623283],
                    [-2.65028866, -1.23856217, -0.51739315],
                    [-2.86540149, -0.51680531, -0.7183625],
                    [-1.98461469, -0.42890191, -1.70646586]])

def roc(y_true, y_score, n_classes,num,name):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    config = {
        'savefig.dpi' : 300,
        'font.serif':['Times New Roman']
    }
    plt.figure()
    #plt.rcParams['savefig.dpi'] = 300
    plt.rcParams.update(config)
    # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
    #                                            ''.format(roc_auc["micro"]), color='deeppink', linestyle=':',linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"], label='macro-average(area = {0:0.2f})'
                                               ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=cm.GnBu(0.08 + i / n_classes), lw=lw,
                 label='class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Blood Glucose Classification')
    plt.legend(loc="lower right",fontsize=8)
    plt.show()
    #plt.savefig('plots/{}/roc/figure_{}.jpg'.format(name,num))

# plot class==2
# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


# def confusion_matrix(preds, labels, conf_matrix):
#     preds = torch.argmax(preds, 1)  # 返回每一行张量最大值的索引
#     labels = torch.argmax(labels, 1)
#     for p, t in zip(preds, labels):
#         conf_matrix[p, t] += 1
#     return conf_matrix


def score(y_predict, y_true):  # 此处的predict，true 都是one-hot编码的类别
    pred = y_predict.detach().numpy()
    true = y_true.detach().numpy()
    j = 0
    true_list = []
    predict_list = []
    for i in range(true.shape[0]):
        true_list.append(true[i].argmax())
        predict_list.append(pred[i].argmax())
    acc = accuracy_score(true_list, predict_list)
    # for i in range(len(predict_list)):
    #     if true_list[i] == predict_list[i]:
    #         j = j + 1
    # return j / len(predict_list)
    return acc


def precision(y_predict, y_true):  # 此处的predict，true 都是one-hot编码的类别
    pred = y_predict.detach().numpy()
    true = y_true.detach().numpy()
    j = 0
    true_list = []
    predict_list = []
    for i in range(true.shape[0]):
        true_list.append(true[i].argmax())
        predict_list.append(pred[i].argmax())
    prec = precision_score(true_list, predict_list, average='macro')
    return prec


def recall(y_predict, y_true):  # 此处的predict，true 都是one-hot编码的类别
    pred = y_predict.detach().numpy()
    true = y_true.detach().numpy()
    j = 0
    true_list = []
    predict_list = []
    for i in range(true.shape[0]):
        true_list.append(true[i].argmax())
        predict_list.append(pred[i].argmax())
    rec = recall_score(true_list, predict_list, average='macro')
    return rec


# if __name__ == '__main__':
#     roc(y_test, y_score, n_classes)