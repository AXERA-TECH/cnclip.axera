import numpy as np
import torch
import numpy as np
import logging
import time
import torch
import os
from tqdm import tqdm
import sys 
sys.path.append('/data/baizanzhou/project/inner/cnclip/')
from cali_data.imagenet_dataset import ImagenetDataset, imagenet_classes, imagenet_templates
import onnx
import onnxruntime as ort
import numpy as np
import random
import cn_clip.clip as clip
import torch



for i, classname in enumerate(imagenet_classes):
    if i>=64:
        break

    idx = random.randint(0, 32)

    texts = [imagenet_templates[idx].format(classname)]
    # format with class
    # inputs = processor(text=texts, images=None, padding="max_length", return_tensors="pt")
    texts = clip.tokenize(texts).to("cpu")
    s_path = f"dataset/bert_cali/{idx}.npy"
    print("save: ", s_path, texts.shape)
    np.save(s_path, texts.numpy())
