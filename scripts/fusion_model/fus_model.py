import numpy as np
import pandas as pd
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision
import pydicom


from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   

class FusionModel(nn.Module):
    def __init__(self, model_container, num_classes=len(classes), features=feats_to_keep, classes=classes, include_nlp=True):
        super(FusionModel, self).__init__()
        self.model_container = model_container
        self.classes = classes
        self.num_classes = num_classes
        self.features = features
        self.include_nlp = include_nlp
        self.num_inputs = num_classes * 3 if self.include_nlp == True else num_classes * 2
        
    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = torch.cat((x1, x2, x3), dim=0)
            x = self.model_container.fusion_model(x)
        else:
            x = torch.cat((x1, x2), dim=0)
            #self.fusion_layer.weight = nn.Parameter(self.model_container.fusion_model.fusion_layer.weight)
            #not implemented currently
        return x