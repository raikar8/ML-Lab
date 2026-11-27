"""
title: Knowledge-distillation loss with temperature and T² scaling
week: 23
day: 3
level: easy
time: 20
tags: pytorch, distillation
core: yes

`kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5)` =
`alpha·CE(student, labels) + (1−alpha)·T²·KL(softmax(teacher/T) ‖ softmax(student/T))` with the KL averaged over the
batch (`F.kl_div(log_softmax(s/T), log_softmax(t/T), log_target=True, reduction="batchmean")`). The teacher must not
receive gradients.
"""
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    hard = F.cross_entropy(student_logits, labels)
    t = teacher_logits.detach()
    soft = F.kl_div(F.log_softmax(student_logits / T, -1), F.log_softmax(t / T, -1), log_target=True,
                    reduction="batchmean")
    return alpha * hard + (1 - alpha) * T * T * soft
