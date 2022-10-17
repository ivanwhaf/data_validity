"""
Teacher Model: EMA Model
"""


class EMA(object):
    """
    Mean Teacher Model(EMA)
    """

    def __init__(self, student_model, beta):
        """
        :param student_model: 学生模型（当前网络模型）
        :param beta: β值，用于调整滑动平均的比例
        """
        self.model = student_model
        self.beta = beta  # β
        self.teacher = {}  # mean teacher
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.teacher[name] = param.data.clone()

    def update(self):
        """
        在每次更新完学生模型后更新ema教师模型
        :return: None
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                average = self.beta * self.teacher[name] + (1.0 - self.beta) * param.data
                self.teacher[name] = average.clone()

    def apply_teacher(self):
        """
        每个epoch训练完后，在evaluate或test之前，应用教师模型
        :return: None
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.teacher[name]

    def restore_student(self):
        """
        evaluate或test之后恢复学生模型，继续训练
        :return: None
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
