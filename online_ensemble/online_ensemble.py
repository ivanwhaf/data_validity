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

    def sort_and_resample(self, outputs, scores, batch_size, lower_ratio, upper_ratio):
        """

        :param outputs: 网络的输出（当前批数据）
        :param scores: 当前批数据的分数，为一个列表，索引要和outputs一一对应
        :param batch_size: 当前批数据的大小
        :param lower_ratio:
        :param upper_ratio:
        :return:
        """
        indices = np.argsort(scores)
        lower_bound = int(batch_size * lower_ratio)
        upper_bound = int(batch_size * upper_ratio)
        indices = indices[lower_bound:upper_bound]
        outputs = outputs[indices]
        return outputs

    def sort_and_reweight(self, outputs, scores):
        # 根据分数对当前批次数据重加权
        # 返回重加权的权重
        indices = np.argsort(scores)
        weights = np.array(len(scores))

        return outputs, weights
