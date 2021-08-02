import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms, models
from PIL import Image

tfms = transforms.Compose([transforms.ToTensor(),
                           transforms.Resize((224, 224)),
                           transforms.Normalize(mean, std)])

def predict(img):
    img = tfms(Image.open(x).convert('RGB'))
    img = img.reshape(1, 3, 224, 224)
    with torch.no_grad():
        ps = model.forward(img)
    top_p, _ = ps.topk(1, dim = 1)
    return top_p
