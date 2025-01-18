#!/usr/bin/env python3

import torch


def get_rotation_between(x, y):
    xy = torch.dot(x, y)
    xx = torch.dot(x, x)
    yy = torch.dot(y, y)
    return torch.acos(xy / torch.sqrt(xx) / torch.sqrt(yy)) / 2 / torch.pi
