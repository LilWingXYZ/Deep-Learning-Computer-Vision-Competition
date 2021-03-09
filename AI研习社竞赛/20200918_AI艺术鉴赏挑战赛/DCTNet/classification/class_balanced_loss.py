# import numpy as np
# import torch
# import torch.nn.functional as F

# def focal_loss(logits, labels, alpha, gamma):
#     """Compute the focal loss between `logits` and the ground truth `labels`.

#     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
#     where pt is the probability of being classified to the true class.
#     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

#     Args:
#       logits: A float tensor of size [batch, num_classes].
#       labels: A float tensor of size [batch, num_classes].
#       alpha: A float tensor of size [batch_size]
#         specifying per-example weight for balanced cross entropy.
#       gamma: A float scalar modulating loss from hard and easy examples.

#     Returns:
#       focal_loss: A float32 scalar representing normalized total loss.
#     """
#     bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

#     if gamma == 0.0:
#         modulator = 1.0
#     else:
#         modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

#     loss = modulator * bce_loss
#     #print("alpha:"+str(alpha))
#     weighted_loss = alpha * loss
#     loss = torch.sum(weighted_loss)
#     loss /= torch.sum(labels)
#     return loss

# class ClassBalancedLoss(torch.nn.Module):
#     def __init__(self, samples_per_class=None, beta=0.99, gamma=2, loss_type="focal"):
#         super(ClassBalancedLoss, self).__init__()
#         if loss_type not in ["focal", "sigmoid", "softmax"]:
#             loss_type = "focal"
#         if samples_per_class is None:
#             num_classes = 5000
#             samples_per_class = [1] * num_classes
#             print("samples_per_class is None!")
#         effective_num = 1.0 - np.power(beta, samples_per_class)
#         weights = (1.0 - beta) / np.array(effective_num)
#         self.constant_sum = len(samples_per_class)
#         weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)
#         self.class_weights = weights
#         self.beta = beta
#         self.gamma = gamma
#         self.loss_type = loss_type

#     def update(self, samples_per_class):
#         if samples_per_class is None:
#             return
#         effective_num = 1.0 - np.power(self.beta, samples_per_class)
#         weights = (1.0 - self.beta) / np.array(effective_num)
#         self.constant_sum = len(samples_per_class)
#         weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)
#         self.class_weights = weights

#     def forward(self, x, y):
#         _, num_classes = x.shape
#         labels_one_hot = F.one_hot(y, num_classes).float()
#         weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y)
#         weights = weights.unsqueeze(1)
#         if self.loss_type == "focal":
#             cb_loss = focal_loss(x, labels_one_hot, weights, self.gamma)
#         elif self.loss_type == "sigmoid":
#             cb_loss = F.binary_cross_entropy_with_logits(x, labels_one_hot, weights)
#         else:  # softmax
#             pred = x.softmax(dim=1)
#             cb_loss = F.binary_cross_entropy(pred, labels_one_hot, weights)
#         return cb_loss

# def test():
#     torch.manual_seed(123)
#     batch_size = 10
#     num_classes = 5
#     x = torch.rand(batch_size, num_classes)
#     y = torch.randint(0, 5, size=(batch_size,))
#     print(x)
#     samples_per_class = [1, 2, 3, 4, 5]
#     loss_type = "focal"
#     loss_fn = ClassBalancedLoss(samples_per_class, loss_type=loss_type)
#     loss = loss_fn(x, y)
#     print(loss)

# if __name__ == '__main__':
#     test()
"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

import numpy as np
import torch
import torch.nn.functional as F


class ClassBalancedLoss(torch.nn.Module):
    def __init__(self,
                 samples_per_class=None,
                 beta=0.99,
                 gamma=2,
                 loss_type="focal"):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

    def forward(self, x, y):
        loss = CB_loss(y, x, self.samples_per_class, 49, self.loss_type,
                       self.beta, self.gamma)
        return loss


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits,
                                                target=labels,
                                                reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits -
                              gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta,
            gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    labels = labels.cuda()
    logits = logits.cuda()

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().cuda()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = (weights.repeat(labels_one_hot.shape[0], 1).cuda() * labels_one_hot.cuda()).cuda()
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels_one_hot,
                                                     weights=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred,
                                         target=labels_one_hot,
                                         weight=weights)
    return cb_loss


if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10, no_of_classes).float()
    labels = torch.randint(0, no_of_classes, size=(10, ))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2, 3, 1, 2, 2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,
                      loss_type, beta, gamma)
    print(cb_loss)
