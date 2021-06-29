import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CWLossFunc(nn.Module):
    def __init__(self):
        super(CWLossFunc, self).__init__()

    def forward(self, input, target):
        correct_logit = input[0][target]
        tem_tensor = torch.zeros(input.shape[-1]).cuda()
        tem_tensor[target] = -10000
        wrong_logit = input[0][torch.argmax(input[0] + tem_tensor)]
        # if correct<wrong then classify false so loss>-50
        # when loss is -20, wrong_logit may be very high
        return -F.relu(correct_logit - wrong_logit + 50)


def nattack_loss(inputs, targets, device):
    batch_size = inputs.shape[0]
    losses = torch.zeros(batch_size).to(device)
    for i in range(batch_size):
        target = targets[i]
        correct_logit = inputs[i][target]
        tem_tensor = torch.zeros(inputs.shape[-1]).to(device)
        tem_tensor[target] = -10000
        wrong_logit = inputs[i][torch.argmax(inputs[i] + tem_tensor)]
        losses[i] = wrong_logit - correct_logit
    return losses


def is_adversarial(model:nn.Module, x:torch.Tensor, y:torch.Tensor, mean, std):
    if x.dim() == 3:
        x.unsqueeze_(0)
    x_normalize = (x - mean) / std
    out = model(x_normalize)
    pred = torch.argmax(out)
    return pred != y


def scale(vec, dst_low, dst_high, src_low, src_high):
    k = (dst_high - dst_low) / (src_high - src_low)
    b = dst_low - k * src_low
    return k * vec + b


def scale_to_tanh(vec):
    return scale(vec, 1e-6 - 1, 1 - 1e-6, 0.0, 1.0)


def clip_eta(eta, distance_metric, eps):
    if distance_metric == 'l_inf':
        eta = torch.clamp(eta, -eps, eps)
    elif distance_metric == 'l_2':
        norm = torch.max(torch.tensor([1e-12, torch.norm(eta)]))
        factor = torch.min(torch.tensor([1, eps/norm]))
        eta = eta * factor
    else:
        raise NotImplementedError
    return eta


def input_diversity(image, prob, low=200, high=224):
    if random.random() > prob:
        return image
    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
    h_remain = high - rnd
    w_remain = high - rnd
    pad_top = random.randint(0, h_remain)
    pad_bottom = h_remain - pad_top
    pad_left = random.randint(0, h_remain)
    pad_right = w_remain - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded

