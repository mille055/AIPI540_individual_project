import pydicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image, ImageDraw
from glob import glob

import sys
sys.path.append("../scripts/")
from  process_tree import Processor 
from  fusion_model.fus_model import FusionModel # Import your machine learning model function
from fusion_model.fus_inference import  get_fusion_inference_from_file
from  config import *
from utils import *
from  model_container import ModelContainer



# Function to check if the image has been processed and return the value in the DICOM tag (0010, 1010)
def check_prediction_tag(dcm_data):
    if (0x0011, 0x1010) in dcm_data:
        prediction =  abd_label_dict[str(dcm_data[0x0011, 0x1010].value)]['short']  # this gets the numeric label written into the DICOM and converts to text description
        # if there are submodel predictions
        prediction_meta = None
        prediction_cnn = None
        prediction_nlp = None
        if (0x0011, 0x1012) in dcm_data:
            substring = dcm_data[0x0011, 0x1012].value
            sublist = substring.split(',')
            try:
                prediction_meta = abd_label_dict[sublist[0]]['short']
                prediction_cnn = abd_label_dict[sublist[1]]['short']
                prediction_nlp = abd_label_dict[sublist[2]]['short']
            except Exception as e:
                pass
        return prediction, prediction_meta, prediction_cnn, prediction_nlp
    else:
        return None, None, None, None
    

@st.cache_resource
def load_dicom_data(folder):
    # function for getting the dicom images within the subfolders of the selected root folder
    # assumes the common file structure of folder/patient/exam/series/images
    data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".dcm"):
                try:
                    dcm_file_path = os.path.join(root, file)
                    dcm_data = pydicom.dcmread(dcm_file_path)
                    label, _, _, _ = check_prediction_tag(dcm_data)
                    data.append(
                        {
                            "patient": dcm_data.PatientName,
                            "exam": dcm_data.StudyDescription,
                            "series": dcm_data.SeriesDescription,
                            "file_path": dcm_file_path,
                            "label": label
                        }
                    )
                except Exception as e:
                    with st.exception("Exception"):
                        st.error(f"Error reading DICOM file {file}: {e}")

    return pd.DataFrame(data)


# for adjusting the W/L of the displayed image
def apply_window_level(image, window_center, window_width):
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    image = np.clip(image, min_value, max_value)
    return image

def normalize_array(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max != arr_min:
        return (arr - arr_min) * 255 / (arr_max - arr_min)
    else:
        return 0
    
def get_single_image_inference(image_path, model_container, fusion_model):
    # single_image_df = pd.DataFrame.from_dicoms([image_path])
    # print('getting file', single_image_df.fname.iloc[0])
    # single_image_df, _ = preprocess(single_image_df, model_container.metadata_scaler)
    scaler = model_container.metadata_scaler
    
    img_df = pd.DataFrame.from_dicoms([image_path])
    img_df, _ = preprocess(img_df, scaler)

    predicted_series_class, predicted_series_confidence, submodel_df = fusion_model.get_fusion_inference(img_df)
    
    predicted_type = abd_label_dict[str(predicted_series_class)]['short'] #abd_label_dict[str(predicted_series_class)]['short']
    prediction_meta = abd_label_dict[str(submodel_df['meta_preds'].values.tolist()[0])]['short'] #abd_label_dict[str(submodel_df['meta_preds'])]['short']
    cnn_prediction = abd_label_dict[str(submodel_df['pixel_preds'].values.tolist()[0])]['short']
    nlp_prediction = abd_label_dict[str(submodel_df['nlp_preds'].values.tolist()[0])]['short']
    
    return predicted_type, prediction_meta, cnn_prediction, nlp_prediction