from attacker import DIM, MIM, FGSM, PGD, DeepFool, Nattack, BIM, TIDIM
import random
import torch
import torchvision

net = torchvision.models.resnet50(pretrained=True)
net.eval()
inputs = torch.rand(3, 224, 224)
method = DIM(eps=8.0 / 255, steps=20, momentum=1, low=320, high=384)
y = torch.tensor(random.randint(0, 1000))
print(y)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
adv = method.generate(net, inputs, y, mean, std)
print(adv.shape)


