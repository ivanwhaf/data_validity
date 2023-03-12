import numpy as np


class OnlineEnsemble(object):
    """
    <在线集成>方法工具类
    """

    def __init__(self):
        self.ensemble_models = []  # 预训练完的集成基模型
        self.n = 0  # 集成模型数量

    def load_pretrain_ensemble_models(self, ensemble_models):
        """
        加载预训练好的基模型

        :param ensemble_models: 预训练好的基模型，是一个列表
        :return: None
        """

        self.ensemble_models = ensemble_models
        self.n = len(self.ensemble_models)

    def online_inference(self, inputs):
        """
        在线推理，用基模型对当前批数据进行推理，并返回结果

        :param inputs: 当前批数据（输入）
        :return: 返回基模型对当前批数据的推理结果，outputs_lst是一个列表

        Examples
        --------
        一个常见训练的流程:

        >>> oe = OnlineEnsemble(ensemble_models)
        >>> for idx,(inputs, labels) in enumerate(train_loader):
        >>>     inputs = inputs.to(device)
        >>>     outputs_lst = oe.online_inference(inputs)
        >>>     ...
        """

        outputs_lst = []
        for model in self.ensemble_models:
            outs = model(inputs)
            outputs_lst.append(outs)
        return outputs_lst

    def sort_and_resample(self, scores, n, lower_ratio, upper_ratio):
        """
        根据分数对当前批次数据重采样

        :param scores: 当前批数据的分数，为一个列表，索引要和outputs一一对应
        :param n: 当前批数据的大小
        :param lower_ratio: 重采样下界的比例，如去除scores较低的5%数据，lower_ratio应为0.05
        :param upper_ratio: 重采样上界的比例，如去除scores较高的5%数据，lower_ratio应为0.95
        :return: 返回重采样后的indices
        """
        indices = np.argsort(scores)
        lower_bound = int(n * lower_ratio)
        upper_bound = int(n * upper_ratio)
        indices = indices[lower_bound:upper_bound]
        return indices

    def sort_and_reweight(self, scores, n, lower_ratio, upper_ratio):
        """
        根据分数对当前批次数据重加权

        :param scores: 当前批数据的分数，为一个列表，索引要和outputs一一对应
        :param n: 当前批数据的大小
        :param lower_ratio: 重加权的下界的比例，如重加权scores较低的5%数据，lower_ratio应为0.05
        :param upper_ratio: 重加权的上界的比例，如重加权scores较高的5%数据，lower_ratio应为0.95
        :return: 返回重加权的权重
        """
        indices = np.argsort(scores)
        lower_bound = int(n * lower_ratio)
        upper_bound = int(n * upper_ratio)

        diff1 = int(lower_bound / 10)
        diff2 = int((n - upper_bound) / 10)

        # 线性加权：权重从0.0递增至0.9（从scores两端向内）
        weights = np.ones(len(scores))
        re_weights = np.ones(len(scores))
        for i in range(10):
            weights[i * diff1:(i + 1) * diff1] = 0.1 * i
            weights[n - (i + 1) * diff2:n - i * diff2] = 0.1 * i

        for i in range(len(indices)):
            idx = indices[i]
            re_weights[idx] = weights[i]

        return re_weights
