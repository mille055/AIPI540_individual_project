###### For configuration of the training or inference of the models ######
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd


  ### locations of assets ###
file_dict = {
'img_data_dir_colab':  '/content/gdrive/MyDrive/WW_MRI_abd2/split/',
'img_data_dir_local': 'volumes/cm7/Abdominal_MRI_dataset_split/',
'txt_data_dir':  '../data/',
'test_datafile': '../data/X_test02282023.pkl',
'train_datafile': '../data/X_train02282023.pkl',
'dataset_file': './stored_assets/dataset.pkl',
'train_csv_file': 'trainfiles.csv',
'test_csv_file': 'testfiles.csv',
'metadata_model_file':  './stored_assets/metadata_model.pkl',
'pixel_model_file': './stored_assets/pixel_model_file.pkl',
'series_description_model_file': './stored_assets/series_description_model_file.pkl',
'labels_file': '../data/cmm_labels.txt'
}

#validation split
val_list =  [41, 84, 14, 25, 76, 47,62,0,55,63,101,18,81,3,4,95,66] #using same train/val/test split as in the original split based on the metadata classifier
random_seed = 42
train_val_split_percent = 0.2

#text model 
sentence_encoder = 'all-MiniLM-L6-v2'
series_description_column = 'SeriesDescription_x'
text_label = 'ap_label_code'
new_dict = {}

#metadata feature list
feats = ['MRAcquisitionType', 'AngioFlag', 'SliceThickness', 'RepetitionTime',
       'EchoTime', 'EchoTrainLength', 'PixelSpacing', 'ContrastBolusAgent',
       'InversionTime', 'DiffusionBValue', 'seq_E', 'seq_EP', 'seq_G',
       'seq_GR', 'seq_I', 'seq_IR', 'seq_M', 'seq_P', 'seq_R', 'seq_S',
       'seq_SE', 'var_E', 'var_K', 'var_MP', 'var_MTC', 'var_N', 'var_O',
       'var_OSP', 'var_P', 'var_S', 'var_SK', 'var_SP', 'var_SS', 'var_TOF',
       'opt_1', 'opt_2', 'opt_A', 'opt_ACC_GEMS', 'opt_B', 'opt_C', 'opt_D',
       'opt_E', 'opt_EDR_GEMS', 'opt_EPI_GEMS', 'opt_F', 'opt_FAST_GEMS',
       'opt_FC', 'opt_FC_FREQ_AX_GEMS', 'opt_FC_SLICE_AX_GEMS',
       'opt_FILTERED_GEMS', 'opt_FR_GEMS', 'opt_FS', 'opt_FSA_GEMS',
       'opt_FSI_GEMS', 'opt_FSL_GEMS', 'opt_FSP_GEMS', 'opt_FSS_GEMS', 'opt_G',
       'opt_I', 'opt_IFLOW_GEMS', 'opt_IR', 'opt_IR_GEMS', 'opt_L', 'opt_M',
       'opt_MP_GEMS', 'opt_MT', 'opt_MT_GEMS', 'opt_NPW', 'opt_P', 'opt_PFF',
       'opt_PFP', 'opt_PROP_GEMS', 'opt_R', 'opt_RAMP_IS_GEMS', 'opt_S',
       'opt_SAT1', 'opt_SAT2', 'opt_SAT_GEMS', 'opt_SEQ_GEMS', 'opt_SP',
       'opt_T', 'opt_T2FLAIR_GEMS', 'opt_TRF_GEMS', 'opt_VASCTOF_GEMS',
       'opt_VB_GEMS', 'opt_W', 'opt_X', 'opt__', 'type_ADC', 'type_DIFFUSION', 'type_DERIVED']


column_lists = {
    'keep': [
        'fname',
        # Patient info
        'PatientID',
        # Study info
        'StudyInstanceUID',
        'StudyID',
        # Series info
        'SeriesInstanceUID',
        'SeriesNumber',
        'SeriesDescription',
        'AcquisitionNumber',
        # Image info and features
        'InstanceNumber',
        'ImageOrientationPatient',
        'ScanningSequence',
        'SequenceVariant',
        'ScanOptions',
        'MRAcquisitionType',
        'AngioFlag',
        'SliceThickness',
        'RepetitionTime',
        'EchoTime',
        'EchoTrainLength',
        'PixelSpacing',
        'ContrastBolusAgent',
        'InversionTime',
        'DiffusionBValue',
        'ImageType',
        # Labels
        'plane',
        'seq_label',
        'contrast'],

    'dummies': [
        'ScanningSequence',
        'SequenceVariant',
        'ScanOptions',
        'ImageType'],

    'd_prefixes': [
        'seq',
        'var',
        'opt',
        'type'],

    'binarize': [
        'MRAcquisitionType',
        'AngioFlag',
        'ContrastBolusAgent',
        'DiffusionBValue'],

    'rescale': [
        'SliceThickness',
        'RepetitionTime',
        'EchoTime',
        'EchoTrainLength',
        'PixelSpacing',
        'InversionTime'],

    'dicom_cols': [
        'PatientID',
        # Study info
        'StudyInstanceUID',
        'StudyID',
        'StudyDescription', # to filter on "MRI BRAIN WITH AND WITHOUT CONTRAST" in some cases
        'Manufacturer',
        'ManufacturerModelName',
        'MagneticFieldStrength',
        # Series info
        'SeriesInstanceUID',
        'SeriesNumber',
        'SeriesDescription', # needed for labeling series
        'SequenceName', # may be used for labeling series
        'BodyPartExamined', # to filter on "HEAD" or "BRAIN"
        'AcquisitionNumber',
        # Image info and features
        'InstanceNumber', # i.e. image number
        'SOPClassUID', # to filter on "MR Image Storage"
        'ImageOrientationPatient', # to calculate slice orientation (e.g. axial, coronal, sagittal)
        'EchoTime',
        'InversionTime',
        'EchoTrainLength',
        'RepetitionTime',
        'TriggerTime',
        'SequenceVariant',
        'ScanOptions',
        'ScanningSequence',
        'MRAcquisitionType',
        'ImageType',
        'PixelSpacing',
        'SliceThickness',
        'PhotometricInterpretation',
        'ContrastBolusAgent',
        'AngioFlag', 
        'DiffusionBValue']}

# Data cropping and normalization, also converts single channel to 3 channel for the model
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}

# converts numeric labels to textual descriptors ###
abd_label_dict = {
    '1': {
        'long': 'Anythingelse',
        'short': 'other',
        'plane': 'other',
        'contrast': 'other'
    },
    '2': {
        'long': 'Arterial T1w',
        'short': 'arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '3': {
        'long': 'Early Arterial T1w',
        'short': 'early_arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '4': {
        'long': 'Late Arterial T1w',
        'short': 'late_arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '5': {
        'long': 'Arterial Subtraction',
        'short': 'arterial_sub',
        'plane': 'ax',
        'contrast': '1'
    },
    '6': {
        'long': 'Coronal Late Dynamic T1w',
        'short': 'dynamic_late',
        'plane': 'cor',
        'contrast': '1'
    },
    '7': {
        'long': 'Coronal T2w',
        'short': 't2',
        'plane': 'cor',
        'contrast': '0'
    },
    '8': {
        'long': 'Axial DWI',
        'short': 'dwi',
        'plane': 'ax',
        'contrast': '0'
    },
    '9': {
        'long': 'Axial T2w',
        'short': 't2',
        'plane': 'ax',
        'contrast': '0'
    },
    '10': {
        'long': 'Coronal DWI',
        'short': 'dwi',
        'plane': 'cor',
        'contrast': '0'
    },
    '11': {
        'long': 'Fat Only',
        'short': 'dixon_fat',
        'plane': 'ax',
        'contrast': '0'
    },
    '12': {
        'long': 'Axial Transitional_Hepatocyte T1w',
        'short': 'hepatobiliary',
        'plane': 'ax',
        'contrast': '1'
    },
    '13': {
        'long': 'Coronal Transitional_Hepatocyte T1w',
        'short': 'hepatobiliary',
        'plane': 'cor',
        'contrast': '1'
    },
    '14': {
        'long': 'Axial In Phase',
        'short': 'in_phase',
        'plane': 'ax',
        'contrast': '0'
    },
    '15': {
        'long': 'Coronal In Phase',
        'short': 'in_phase',
        'plane': 'cor',
        'contrast': '0'
    },
    '16': {
        'long': 'Axial Late Dyanmic T1w',
        'short': 'dynamic_equilibrium',
        'plane': 'ax',
        'contrast': '1'
    },
    '17': {
        'long': 'Localizers',
        'short': 'loc',
        'plane': 'unknown',
        'contrast': '0'
    },
    '18': {
        'long': 'MRCP',
        'short': 'mrcp',
        'plane': 'cor',
        'contrast': '0'
    },
    '19': {
        'long': 'Axial Opposed Phase',
        'short': 'opposed_phase',
        'plane': 'ax',
        'contrast': '0'
    },
    '20': {
        'long': 'Coronal Opposed Phase',
        'short': 'opposed_phase',
        'plane': 'cor',
        'contrast': '0'
    },
    '21': {
        'long': 'Proton Density Fat Fraction',
        'short': 'fat_quant',
        'plane': 'ax',
        'contrast': '0'
    },
    '22': {
        'long': 'Water Density Fat Fraction',
        'short': 'water_fat_quant',
        'plane': 'ax',
        'contrast': '0'
    },
    '23': {
        'long': 'Portal Venous T1w',
        'short': 'portal_venous',
        'plane': 'ax',
        'contrast': '1'
    },
    '24': {
        'long': 'Coronal Precontrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'cor',
        'contrast': '0'
    },
    '25': {
        'long': 'Axial Precontrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax',
        'contrast': '0'
    },
    '26': {
        'long': 'R*2',
        'short': 'r_star_2',
        'plane': 'ax',
        'contrast': '0'
    },
    '27': {
        'long': 'Axial Steady State Free Precession',
        'short': 'ssfse',
        'plane': 'ax',
        'contrast': '0'
    },
    '28': {
        'long': 'Coronal Steady State Free Precession',
        'short': 'ssfse',
        'plane': 'cor',
        'contrast': '1'
    },
    '29': {
        'long': 'Venous Subtraction',
        'short': 'venous_sub',
        'plane': 'ax',
        'contrast': '1'
    },
    '0': {
        'long': 'Axial ADC',
        'short': 'adc',
        'plane': 'ax',
        'contrast': '0'
    },
     '30': {
        'long': 'Axial Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax',
        'contrast': '1'
    },
    '31': {
        'long': 'Coronal Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'cor',
        'contrast': '1'
    },
    '32': {
        'long': 'Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax/cor',
        'contrast': '1'
    } }
