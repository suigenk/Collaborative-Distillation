from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self, contentPath, stylePath, texturePath, c_size=0, s_size=0, picked_content_mark=".", picked_style_mark=".", synthesis=False, debug=False):
      super(Dataset,self).__init__()
      self.content_size = c_size
      self.style_size = s_size
      self.synthesis = synthesis
      self.debug = debug
      if synthesis:
        self.texturePath = texturePath
        self.texture_image_list = [x for x in listdir(texturePath) if is_image_file(x)]
      else:
        self.contentPath = contentPath
        self.stylePath   = stylePath
        content_imgs = [x for x in listdir(contentPath) if is_image_file(x) and picked_content_mark in x]
        style_imgs   = [x for x in listdir(stylePath)   if is_image_file(x) and picked_style_mark   in x]
        pairs = [[c, s] for c in content_imgs for s in style_imgs]
        self.content_image_list = list(np.array(pairs)[:, 0])
        self.style_image_list = list(np.array(pairs)[:, 1])

        if self.debug:
          self.content_image_list = self.content_image_list[:5]

        print("# of images", len(self.content_image_list))

    def __getitem__(self, index):
      if not self.synthesis: # style transfer
        contentImgPath = os.path.join(self.contentPath, self.content_image_list[index])
        styleImgPath = os.path.join(self.stylePath, self.style_image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)
        if self.content_size:
          contentImg = transforms.Resize(self.content_size)(contentImg)
        if self.style_size:
          styleImg = transforms.Resize(self.style_size)(styleImg)
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        imname = self.content_image_list[index].split(".")[0] + "_" + self.style_image_list[index].split(".")[0] + ".png"
        return contentImg.squeeze(0), styleImg.squeeze(0), imname

      
      else: # texture synthesis
        textureImgPath = os.path.join(self.texturePath, self.texture_image_list[index])
        textureImg = default_loader(textureImgPath)
        if self.style_size:
          w, h = textureImg.size
          if w > h:
            neww = self.style_size
            newh = int(h * neww / w)
          else:
            newh = self.style_size
            neww = int(w * newh / h)
          textureImg = textureImg.resize((neww,newh))
        w, h = textureImg.size
        textureImg = transforms.ToTensor()(textureImg)
        contentImg = torch.rand_like(textureImg)
        return contentImg.squeeze(0), textureImg.squeeze(0), self.texture_image_list[index].split(".")[0] + ".jpg"

    def __len__(self):
        return len(self.texture_image_list) if self.synthesis else len(self.content_image_list)