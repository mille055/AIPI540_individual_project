from __future__ import print_function
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk, spacy
import sklearn
from sklearn.model_selection import train_test_split
import os
import os.path
import glob
import pydicom

import sys
from random import shuffle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, plot_confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import copy
from datetime import datetime
import pickle 

from config import file_dict, feats, column_lists, RF_parameters, classes
from config import abd_label_dict, val_list, train_val_split_percent, random_seed, data_transforms
from config import sentence_encoder, series_description_column
from utils import shorten_df, plot_and_save_cm, prepare_df

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
senttrans_model = SentenceTransformer(sentence_encoder, device=device)


def load_text_data(train_csv, val_csv, test_csv):
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    train_df = shorten_df(train_df)
    val_df = shorten_df(val_df)
    test_df = shorten_df(test_df)
    
    train_df = prepare_df(train_df)
    val_df = prepare_df(val_df)
    test_df = prepare_df(test_df)

    return train_df, val_df, test_df



def train_text_log_model(train_data, val_data, test_data, senttrans_model=sentence_encoder):
    X_train = train_data['SeriesDescription']
    y_train = train_data['label']
    
    X_val = val_data['SeriesDescription']
    y_val = val_data['label']
    
    X_test = test_data['SeriesDescription']
    y_test = test_data['label']

    
    X_train_encoded = [senttrans_model.encode(doc) for doc in X_train.to_list()]
    X_val_encoded = [senttrans_model.encode(doc) for doc in X_val.to_list()]
    X_test_encoded = [senttrans_model.encode(doc) for doc in X_test.to_list()]

    # Train a classification model using logistic regression classifier
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train_encoded, y_train)
    
    train_preds = logreg_model.predict(X_train_encoded)
    train_probs = logreg_model.predict_proba(X_train_encoded)
    train_acc = sum(train_preds == y_train) / len(y_train)
    print('Accuracy on the training set is {:.3f}'.format(train_acc))

    ## assess on the val set
    #print('size of X_val_encoded is ', len(X_val_encoded))
    #print('size of y_val is ', len(y_val))
    val_preds = logreg_model.predict(X_val_encoded)
    val_probs = logreg_model.predict_proba(X_val_encoded)
    print('size of preds_val is ', len(val_preds))
    val_acc = sum(val_preds == y_val)/ len(y_val)
    print('Accuracy on the val set is {:.3f}'.format(val_acc))
    
    ## display results on test set
    test_preds = logreg_model.predict(X_test_encoded)
    test_probs = logreg_model.predict_proba(X_test_encoded)
    test_acc = sum(test_preds == y_test) / len(y_test)
    ## display results on test set
    print('Accuracy on the test set is {:.3f}'.format(test_acc))


    #export model
    #txt_model_filename = "../models/text_model"+ datetime.now().strftime('%Y%m%d') + ".st"
    #pickle.dump(logreg_model, open(txt_model_filename, 'wb'))

    return train_preds, train_probs, train_acc, val_preds, val_probs, val_acc, test_preds, test_probs, test_acc, logreg_model



def list_incorrect_text_predictions(ytrue, ypreds, series_desc):
    ytrue = ytrue.tolist()
    ytrue_label = [abd_label_dict[str(x)]['short'] for x in ytrue]
    ypreds = ypreds.tolist()
    ypreds_label = [abd_label_dict[str(x)]['short'] for x in ypreds]
    ylist = zip(series_desc, ytrue, ypreds)
    ylist_label = zip(series_desc,ytrue_label, ypreds_label)
    y_incorrect_list = [x for x in ylist if x[1]!=x[2]]
    y_incorrect_list_label = [x for x in ylist_label if x[1]!=x[2]]
    return y_incorrect_list, y_incorrect_list_label


def get_NLP_inference(model, filenames, device=device, classes=classes):
    
    senttrans_model = SentenceTransformer(sentence_encoder, device=device)
    preds = []
    probs = []

    for filename in filenames:
        print(filename)
        try:
            ds = pydicom.dcmread(filename)
            description = ds.SeriesDescription
            
            description_encoded = senttrans_model.encode(description)

            #print(f'Getting prediction for file {filename} with SeriesDesription label {description} and shape {description_encoded.shape}')

            pred = model.predict(description_encoded.reshape(1, -1))[0]  # Use description_encoded and reshape to a 2D array
            preds.append(pred)
            prob = model.predict_proba(description_encoded.reshape(1, -1))  # Use description_encoded and reshape to a 2D array
            probs.append(prob)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    preds_np = np.array(preds)  # Convert preds to a NumPy array
    probs_np = np.array(probs).squeeze()  # Convert probs to a NumPy array


    return preds_np, probs_np


## test
## csv files
# train_datafile = '../data/trainfiles.csv'
# test_datafile = '../data/testfiles.csv'
# val_datafile = '../data/valfiles.csv'

# train_data, val_data, test_data = load_text_data2(train_datafile, val_datafile, test_datafile)
# train_preds, train_probs, train_acc, val_preds, val_probs, val_acc, test_preds, test_probs, test_acc, logreg_model = train_text_log_model2(train_data, val_data, test_data)
# print(test_preds, test_acc)
# # list, list_label = list_incorrect_text_predictions(y_test, preds_test, series_desc)
# # print(list_label)