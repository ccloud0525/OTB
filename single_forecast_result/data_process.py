import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_all_files_in_directory(directory):
    all_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if file_path.endswith('.csv'):
                all_files.append(file_path)
    return all_files


directory_path = 'chosen_datasets/单变量序列预测160条'
all_files = get_all_files_in_directory(directory_path)
file_name = []
for file_path in all_files:
    file_name.append(os.path.basename(file_path))

file_list = os.listdir('chosen_datasets')
for file in file_list:
    if os.path.isfile(os.path.join('chosen_datasets', file)) and file in file_name:
        os.remove(os.path.join('chosen_datasets', file))

directory_path = 'forth/ori_all_datasets'
all_files = get_all_files_in_directory(directory_path)
all_df = []
for open_file in all_files:
    df = pd.read_csv(open_file)
    all_df.append(df)

dataset_algorithm = {}
file_list = os.listdir('chosen_datasets')
for file in tqdm(file_list):
    if os.path.isfile(os.path.join('chosen_datasets', file)):
        comparison_list = []
        for df in all_df:
            selected_df = df[df['file_name'] == file]
            selected_df['smape'] = selected_df['smape'].astype(str).str.split(';').apply(lambda x: float(x[0]))
            if not selected_df.empty:
                comparison_list.append(selected_df)

        if len(comparison_list) > 0:
            new_comparison_list = sorted(comparison_list, key=lambda x: float(x['smape']))
            new_df = new_comparison_list[0]
            alg = new_df.iloc[0, new_df.columns.get_loc('model_name')]
            dataset_algorithm[file] = alg

directory_path = 'forth/ori_all_datasets'
all_files = get_all_files_in_directory(directory_path)
model_name_list = []

for open_file in all_files:
    data = pd.read_csv(open_file)
    model_name = data['model_name'][0]
    if model_name not in model_name_list:
        model_name_list.append(model_name)

model_name_dict = {
    'FiLM': 0, 'DLinear': 1, 'MICN': 2, 'TimesNet': 3, 'NLinear': 4, 'darts_nbeatsmodel': 5, 'Triformer': 6,
    'darts_tcnmodel': 7, 'PatchTST': 8, 'Crossformer': 9, 'FEDformer': 10, 'Linear': 11, 'darts_nhitsmodel': 12,
    'Nonstationary_Transformer': 13, 'Informer': 14, 'darts_blockrnnmodel': 15, 'darts_statsforecastautoets': 16,
    'darts_autoarima': 17, 'darts_statsforecastautotheta': 18, 'darts_randomforest': 19, 'darts_kalmanforecaster': 20,
    'darts_naivemovingaverage': 21, 'darts_rnnmodel': 22, 'darts_tidemodel': 23, 'darts_naivemean': 24,
    'darts_naivedrift': 25, 'darts_xgbmodel': 26, 'darts_linearregressionmodel': 27, 'darts_naiveseasonal': 28,
    'darts_lightgbmmodel': 29, 'darts_statsforecastautoces': 30
}

dataset_algorithm_one_hot_list = []
for dataset, algorithm in dataset_algorithm.items():
    vector = [0] * 31
    vector[model_name_dict[algorithm]] = 1
    dataset_algorithm_one_hot_list.append((dataset, vector))

np.save('dataset_algorithm.npy', dataset_algorithm_one_hot_list)
