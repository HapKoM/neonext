"""
This code is based on https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
"""
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from functools import partial


# from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

def shear_x(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        Image.BICUBIC, fillcolor=fillcolor
    )

def shear_y(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        Image.BICUBIC, fillcolor=fillcolor
    )

def translate_x(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=fillcolor
    )

def translate_y(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=fillcolor
    )

def color(img, magnitude):
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))

def posterize(img, magnitude):
    return ImageOps.posterize(img, magnitude)

def solarize(img, magnitude):
    return ImageOps.solarize(img, magnitude)

def contrast(img, magnitude):
    return ImageEnhance.Contrast(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )

def sharpness(img, magnitude):
    return ImageEnhance.Sharpness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )

def brigthness(img, magnitude):
    return ImageEnhance.Brightness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )

def autocontrast(img, magnitude):
    return ImageOps.autocontrast(img)

def equalize(img, magnitude):
    return ImageOps.equalize(img)

def invert(img, magnitude):
    return ImageOps.invert(img)


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]

    def __call__(self, img, policy_idx=None):
        if policy_idx is None or not isinstance(policy_idx, int):
            policy_idx = random.randint(0, len(self.policies) - 1)
        else:
            policy_idx = policy_idx % len(self.policies)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    def __init__(
        self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2,
        fillcolor=(128, 128, 128)
    ):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": partial(shear_x, fillcolor=fillcolor),
            "shearY": partial(shear_y, fillcolor=fillcolor),
            "translateX": partial(translate_x, fillcolor=fillcolor),
            "translateY": partial(translate_y, fillcolor=fillcolor),
            "rotate": rotate_with_fill,
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": color,
            "posterize": posterize,
            "solarize": solarize,
            "contrast": contrast,
            "sharpness": sharpness,
            "brightness": brigthness,
            "autocontrast": autocontrast,
            "equalize": equalize,
            "invert": invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img
