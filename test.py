from model import Model
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as fn
import matplotlib
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import cv2


def test():
    model = Model()
    model.eval()
    preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
    img = cv2.imread("content/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 1, facecolor='white')
    ax[0].set_title("Original")
    ax[0].imshow(img)

    imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
    preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])])
    img = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        pr_mask = model(img)
        print(pr_mask.shape)
        pr_mask = pr_mask.softmax(1).squeeze()
        print(pr_mask)

    ax[1].imshow(pr_mask.numpy())  # just squeeze classes dim, because we have only one class
    ax[1].set_title("Segmentation map")
    plt.tight_layout()
    plt.show()
