# Several methods: FGSM, BIM

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils
from utils import CWLossFunc, nattack_loss


class Attacker:
    def __init__(self, eps: float = 8.0 / 255, clip_min: float = 0.0, clip_max: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.eps = eps
        self.device = device

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)) -> torch.Tensor:
        pass


class FGSM(Attacker):
    def __init__(self, eps, clip_min, clip_max, device: torch.device = torch.device('cpu')):
        super(FGSM, self).__init__(eps, clip_min, clip_max, device)

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx = nx.to(self.device)
        ny = ny.to(self.device)
        model = model.to(self.device)
        nx.requires_grad_(True)

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)
        nx_normalize = (nx - mean) / std

        out = model(nx_normalize)
        loss = F.cross_entropy(out, ny)
        loss.backward()

        x_adv = nx + self.eps * torch.sign(nx.grad.data)
        x_adv.clamp_(self.clip_min, self.clip_max)

        return x_adv.squeeze(0).detach().clone()


class BIM(Attacker):
    def __init__(self, eps, alpha, steps, clip_min=0.0, clip_max=1.0, device=torch.device('cpu')):
        super(BIM, self).__init__(eps=eps, clip_min=clip_min, clip_max=clip_max, device=device)
        self.alpha = alpha
        self.steps = steps

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        model.eval()
        model = model.to(self.device)
        nx = torch.unsqueeze(x, 0).to(self.device)
        ny = torch.unsqueeze(y, 0).to(self.device)
        nx.requires_grad_(True)

        eta = torch.zeros(nx.shape).to(self.device)

        adv_t = nx + eta

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

        for i in range(self.steps):
            adv_normalize = (adv_t - mean) / std
            out = model(adv_normalize)
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.alpha * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()
            adv_t = nx + eta
            adv_t.clamp_(self.clip_min, self.clip_max)

        return adv_t.squeeze(0).detach().clone()


class PGD(Attacker):
    def __init__(self, eps, steps, alpha, random_start: bool = False, loss_func=None, clip_min=0.0, clip_max=1.0,
                 device=torch.device('cpu')):
        super(PGD, self).__init__(eps=eps, clip_min=clip_min, clip_max=clip_max, device=device)
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start
        if loss_func == 'cw':
            print('using cw loss')
            self.loss_func = CWLossFunc()
        else:
            self.loss_func = F.cross_entropy

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        model.eval()
        if self.random_start:
            x = x + (torch.rand(x.shape) * 2 * self.eps - self.eps)

        nx = torch.unsqueeze(x, 0).to(self.device)
        ny = torch.unsqueeze(y, 0).to(self.device)
        model = model.to(self.device)
        nx.requires_grad_(True)

        eta = torch.zeros(nx.shape).to(self.device)
        adv_t = nx + eta

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

        for i in range(self.steps):
            adv_normalize = (adv_t - mean) / std
            out = model(adv_normalize)
            loss = self.loss_func(out, ny)
            loss.backward()

            eta += self.alpha * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()
            adv_t = nx + eta
            adv_t.clamp_(self.clip_min, self.clip_max)

        return adv_t.squeeze(0).detach()


class DeepFool(Attacker):
    def __init__(self, max_iter=50, clip_max=1.0, clip_min=0.0, device=torch.device('cpu')):
        super(DeepFool, self).__init__(clip_max=clip_max, clip_min=clip_min, device=device)
        self.max_iter = max_iter

    def generate(self, model, x, y, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        model.eval()
        nx = torch.unsqueeze(x, 0).to(self.device)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to(self.device)

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

        adv_t = nx + eta
        adv_normalize = (adv_t - mean) / std

        out = model(adv_normalize)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i / np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()

            adv_t = nx + eta
            adv_normalize = (adv_t - mean) / std
            out = model(adv_normalize)
            py = out.max(1)[1].item()
            i_iter += 1

        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)

        return x_adv.detach()


class MIM(Attacker):
    def __init__(self, eps, steps, loss_func=None, clip_max=1.0, clip_min=0.0, momentum=1.0,
                 device=torch.device('cpu')):
        super(MIM, self).__init__(clip_max=clip_max, clip_min=clip_min, eps=eps, device=device)
        self.steps = steps
        self.step_size = eps / steps
        if loss_func == 'cw':
            print('cw loss')
            self.loss_func = CWLossFunc()
        else:
            self.loss_func = F.cross_entropy
        self.momentum = momentum

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)) -> torch.Tensor:
        model.eval()
        nx = torch.unsqueeze(x, 0).to(self.device)
        ny = torch.unsqueeze(y, 0).to(self.device)
        nx.requires_grad_(True)
        eta = torch.zeros(nx.shape).to(self.device)
        adv_t = nx + eta

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)
        g = 0
        for i in range(self.steps):
            adv_normalize = (adv_t - mean) / std
            out = model(adv_normalize)
            loss = self.loss_func(out, ny)
            loss.backward()
            grad = nx.grad.data
            g = self.momentum * g + grad / grad.norm(p=1)
            eta += self.step_size * torch.sign(g)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()

            adv_t = nx + eta
            adv_t.clamp_(self.clip_min, self.clip_max)
        return adv_t.squeeze(0).detach()


class Nattack(Attacker):
    def __init__(self, eps, max_queries, sample_size=100, distance_metric='l_inf', lr=0.02, sigma=0.1, clip_min=0.0,
                 clip_max=1.0, device: torch.device = torch.device('cpu')):
        super(Nattack, self).__init__(eps=eps, clip_min=clip_min, clip_max=clip_max, device=device)
        self.max_queries = max_queries
        self.sample_size = sample_size
        self.distance_metric = distance_metric
        self.lr = lr
        self.sigma = sigma
        self.loss_func = nattack_loss

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        model.eval()

        nx = torch.unsqueeze(x, 0).to(self.device)
        ny = torch.unsqueeze(y, 0).to(self.device)
        shape = nx.shape
        model = model.to(self.device)

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

        with torch.no_grad():
            y = torch.tensor([y] * self.sample_size)
            y = y.to(self.device)
            q = 0
            pert_shape = [3, 32, 32]
            modify = torch.randn(1, *pert_shape).to(self.device) * 0.001

            while q < self.max_queries:
                pert = torch.randn(self.sample_size, *pert_shape).to(self.device)
                modify_try = modify + self.sigma * pert
                modify_try = F.interpolate(modify_try, shape[-2:], mode='bilinear', align_corners=False)
                # 1. add modify_try to z=tanh(x), 2. arctanh(z+modify_try), 3. rescale to [0, 1]
                arctanh_xs = torch.atanh(utils.scale_to_tanh(nx))
                eval_points = 0.5 * (torch.tanh(arctanh_xs + modify_try) + 1)
                eta = eval_points - nx
                eval_points = nx + utils.clip_eta(eta, self.distance_metric, self.eps)

                inputs = (eval_points - mean) / std
                outputs = model(inputs)
                loss = self.loss_func(outputs, y, self.device)
                normalize_loss = (loss - torch.mean(loss)) / (torch.std(loss) + 1e-7)

                q += self.sample_size

                grad = normalize_loss.reshape(-1, 1, 1, 1) * pert
                grad = torch.mean(grad, dim=0) / self.sigma
                # grad.shape : (sample_size, 3, 32, 32) -> (3, 32, 32)
                modify = modify + self.lr * grad
                modify_test = F.interpolate(modify, shape[-2:], mode='bilinear', align_corners=False)

                adv_t = 0.5 * (torch.tanh(arctanh_xs + modify_test) + 1)
                adv_t = nx + utils.clip_eta(adv_t - nx, self.distance_metric, self.eps)

                if utils.is_adversarial(model, adv_t, ny, mean, std):
                    print('image is adversarial, query', q)
                    return adv_t.squeeze(0).detach()
        return adv_t.squeeze(0).detach()


class DIM(Attacker):
    def __init__(self, eps, steps, momentum, clip_min=0.0, clip_max=1.0, device=torch.device('cpu'), low=200, high=224):
        super(DIM, self).__init__(eps=eps, clip_min=clip_min, clip_max=clip_max, device=device)
        self.steps = steps
        self.step_size = eps / steps
        self.momentum = momentum
        self.loss_func = F.cross_entropy
        self.low = low
        self.high = high

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)) -> torch.Tensor:
        model.eval()
        nx = torch.unsqueeze(x, 0).to(self.device)
        ny = torch.unsqueeze(y, 0).to(self.device)
        nx.requires_grad_(True)

        eta = torch.zeros(nx.shape).to(self.device)
        adv_t = nx + eta
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

        g = 0
        for i in range(self.steps):
            adv_diversity = utils.input_diversity(adv_t, prob=0.9, low=self.low, high=self.high)
            adv_normalize = (adv_diversity - mean) / std
            out = model(adv_normalize)
            loss = self.loss_func(out, ny)
            loss.backward()

            gradient = nx.grad.data
            g = self.momentum * g + gradient / gradient.norm(p=1)

            eta += torch.sign(g)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()
            adv_t = nx + eta
            adv_t.clamp_(self.clip_min, self.clip_max)

        return adv_t.squeeze(0).detach()


if __name__ == '__main__':
    import random
    net = torchvision.models.resnet50(pretrained=True)
    net.eval()
    inputs = torch.rand(3,224,224)
    method = DIM(eps=8.0/255, steps=20, momentum=1, low=320, high=384)
    y = torch.tensor(random.randint(0, 1000))
    print(y)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    adv = method.generate(net, inputs, y, mean, std)
    print(adv.shape)

