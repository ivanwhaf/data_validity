import numpy as np


class ForgettingEvents(object):
    """
    <遗忘效应>方法工具类
    """

    def __init__(self, n):
        """
        :param n: length of dataset
        """
        self.forgetting_times = np.zeros(n)
        self.pred_acc_flag = np.zeros(n)

    def record_forgetting(self, pred_labels, labels, indices):
        """
        record forgetting times each epoch

        :param pred_labels: current epoch predict labels
        :param labels: true labels
        :param indices: /
        :return: None
        """
        for i in range(len(pred_labels)):
            if self.pred_acc_flag[indices[i]] == 1:  # prev epoch predict right
                if pred_labels[i] != labels[i]:  # this epoch predict wrong
                    self.forgetting_times[indices[i]] += 1  # forgetting time plus 1
            self.pred_acc_flag[indices[i]] = 1 if pred_labels[i] == labels[i] else 0
