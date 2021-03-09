import torchvision.transforms as transforms
import cv2
import numpy
import random


class random_crop(object):
    def __call__(self, image):

        a = random.uniform(0.5, 0.9)
        b = random.uniform(0.5, 0.9)
        RandomCrop = transforms.RandomCrop(size=(int(a * image.size[1]),
                                                 int(b * image.size[0])))
        random_image = RandomCrop(image)
        return random_image


class resize(object):
    def __call__(self, image):
        Resize = transforms.Resize(size=(100, 50))
        resized_image = Resize(image)
        return resized_image


class random_resized_crop(object):
    def __call__(self, image):
        RSC = transforms.RandomResizedCrop(size=200,
                                           scale=(0.2, 0.5),
                                           ratio=(1, 5))
        rsc_image = RSC(image)
        return rsc_image


class horizontal_flip(object):
    def __call__(self, image):
        HF = transforms.RandomHorizontalFlip()
        hf_image = HF(image)
        return hf_image


class vertical_flip(object):
    def __call__(self, image):
        VF = transforms.RandomVerticalFlip()
        vf_image = VF(image)
        return vf_image


class random_rotation(object):
    def __call__(self, image):
        RR = transforms.RandomRotation(degrees=(10, 80))
        rr_image = RR(image)
        return rr_image


class tocv2(object):
    def __call__(self, image):
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        return img