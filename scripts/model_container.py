import pickle
#import torch

from config import model_paths
from cnn.cnn_inference import load_pixel_model

class ModelContainer:
    def __init__(self):
        self.cnn_model = load_pixel_model(model_paths2['cnn'])
        self.nlp_model = self.load_model(model_paths2['nlp'])
        self.metadata_model = self.load_model(model_paths2['meta'])
        self.metadata_scaler = self.load_model(model_paths2['scaler'])
        
        #fusion model weights
        self.fusion_weights_path = model_paths2['fusion']
        self.partial_fusion_weights_path = model_paths2['fusion_no_nlp']

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    
