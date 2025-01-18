#!/usr/bin/env python3
import torch

from tiny_stories_sae.common.angle import get_rotation_between


def test_rotation():
    """Sanity checks on 2d vectors"""
    x = torch.tensor([1, 1])
    y = torch.tensor([1, 1])
    assert get_rotation_between(x, y) == 0
    flip_y = torch.tensor([-1, -1])
    assert get_rotation_between(x, flip_y) == 0.5
    turned = torch.tensor([1, -1])
    assert get_rotation_between(x, turned) == 0.25


def test_random():
    """Random high-dimensional vectors should be nearly orthogonal"""
    for _ in range(10):
        x = torch.randn(768)
        y = torch.randn(768)
        assert 0.27 >= get_rotation_between(x, y) >= 0.23
