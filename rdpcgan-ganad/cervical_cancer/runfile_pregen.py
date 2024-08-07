import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import supervised_model_training
import re
from DPAE2 import DP_Autoencoder


def find_highest_n(folder_path, pattern):
    # List all files in the specified folder
    try:
        files = os.listdir(f"{folder_path}")
        # print(folder_path)
    except FileNotFoundError:
        # print(os.listdir("results/pre_gen_eps=10_dataset=cervical-cancer_iters=1_k=1_ADBudget=0.1_delta=1e-05_ns=1_lambda=1.0_noise_rate=1.0_batchsize=64"))
        print(f"Folder '{folder_path}' not found.")
        return 0
    # print("Trying to print files here")
    # print(files)
    # Extract all n values from matching files
    n_values = []
    for file_name in files:
        match = pattern.match(file_name)
        if match:
            n_values.append(int(match.group(1)))

    # Return the highest n value or 0 if no matching files were found
    # print(n_values)
    return max(n_values, default=-1)

def final_testing():
    for e in [0.1, 1, 10]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--data_no',
            help='number of generated data',
            default=10000,
            type=int)
        parser.add_argument(
            '--data_dim',
            help='number of dimensions of generated dimension (if random)',
            default=10,
            type=int)
        parser.add_argument(
            '--dataset',
            help='dataset to use',
            default='cervical-cancer',
            type=str)
        parser.add_argument(
            '--noise_rate',
            help='noise ratio on data',
            default=1.0,
            type=float)
        parser.add_argument(
            '--iterations',
            help='number of iterations for handling initialization randomness',
            default=10,
            type=int)
        parser.add_argument(
            '--n_s',
            help='the number of student training iterations',
            default=1,
            type=int)
        parser.add_argument(
            '--batch_size',
            help='the number of batch size for training student and generator',
            default=64,
            type=int)
        parser.add_argument(
            '--k',
            help='the number of teachers',
            default=50,
            type=float)
        parser.add_argument(
            '--epsilon',
            help='Differential privacy parameters (epsilon)',
            default=e,
            type=float)
        parser.add_argument(
            '--delta',
            help='Differential privacy parameters (delta)',
            default=0.00001,
            type=float)
        parser.add_argument(
            '--lamda',
            help='DPGAN noise size',
            default=1.0,
            type=float)
        parser.add_argument(
            '--AD_budget',
            help='the budget for anomaly detection',
            default=0.1,
            type=float)
        parser.add_argument(
            '--epochs',
            help='the number of epochs to train',
            default=50,
            type=int
        )

        if e == 0.1:
            main_epochs = 1
            noise_mult = 10
            ae_noise_mult = 50
            ae_epochs = 1

        elif e == 1:
            main_epochs = 10
            noise_mult = 4
            ae_noise_mult = 10
            ae_epochs = 5
        elif e == 10:
            main_epochs = 30
            noise_mult = 1
            ae_noise_mult = 1.8
            ae_epochs = 10
        else:
            raise Exception('Unimplemented epsilon value')

        args = parser.parse_args()

        data = pd.read_csv('cervical-cancer_csv.csv')
        data = data.replace('?', np.nan)
        data.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], inplace=True, axis=1)
        numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
                        'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)',
                        'STDs (number)']
        categorical_df = ['Smokes', 'Hormonal Contraceptives', 'IUD',
                          'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
                          'STDs:vaginal condylomatosis',
                          'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                          'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
                          'STDs:molluscum contagiosum', 'STDs:AIDS',
                          'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV',
                          'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN',
                          'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology',
                          'Biopsy']
        # Fills the missing values of numeric data columns with mean of the column data.
        for feature in numerical_df:
            # print(feature, '', data[feature].apply(pd.to_numeric, errors='coerce').mean())
            feature_mean = round(data[feature].apply(pd.to_numeric, errors='coerce').mean(), 1)
            data[feature] = data[feature].fillna(feature_mean)
        for feature in categorical_df:
            data[feature] = data[feature].apply(pd.to_numeric,
                                                errors='coerce').fillna(1.0)

        data = MinMaxScaler().fit_transform(data)
        train_ratio = 0.5
        train = np.random.rand(data.shape[0]) < train_ratio
        train_data, test_data = data[train], data[~train]
        data_dim = data.shape[1]


        ad_epsilon = args.epsilon * args.AD_budget


        models = ['logisticregression', 'randomforest', 'gaussiannb', 'bernoullinb',
                      'svmlin', 'Extra Trees', 'LDA', 'AdaBoost', 'Bagging', 'gbm', 'xgb']

        number_of_tests = 10
        folder_name = f"results/pre_gen_eps={args.epsilon}_dataset={args.dataset}_iters={args.iterations}_k={args.k}_ADBudget={args.AD_budget}_delta={args.delta}_ns={args.n_s}_lambda={args.lamda}_noise_rate={args.noise_rate}_batchsize={args.batch_size}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        pattern_results = re.compile(r"results-pategan-(\d+).csv$")

        results_start = find_highest_n(folder_name, pattern_results)

        pattern_synth_data = re.compile(r"synth-data-pategan-(\d+).csv$")

        synth_data_start = find_highest_n(folder_name, pattern_synth_data)

        for n in range(1, number_of_tests + 1):
            print(f"Starting test number: {n}")
            os.system(
                f"venv_directory/python autoencoder_pregen.py "
                f"--n_epochs_pretrain {ae_epochs} --noise_multiplier {ae_noise_mult} --batch_size 16")
            print("Finished with training AE")
            os.system(
                f"venv_directory/python wganprivate_pregen.py "
                f"--classLabel 0.0 --n_epochs {main_epochs} --noise_multiplier {noise_mult} --AD_budget {ad_epsilon} --batch_size 16")

            os.system(
                f"venv_directory/python wganprivate_pregen.py "
                f"--classLabel 1.0 --n_epochs {main_epochs} --noise_multiplier {noise_mult} --AD_budget {ad_epsilon} --batch_size 16")

            os.system(
                f"venv_directory/python evaluate_pregen.py "
                f"--classLabel 0.0 --n_epochs {main_epochs} --noise_multiplier {noise_mult} --batch_size 16")

            os.system(
                f"venv_directory/python evaluate_pregen.py "
                f"--classLabel 1.0 --n_epochs {main_epochs} --noise_multiplier {noise_mult} --batch_size 16")

            # experiment_name = 'uci'
            # exp_path = os.path.expanduser('~/experiments/pytorch/' + experiment_name)



            gen_samples_0 = np.load("synthetic_pregen/synthetic_0.npy", allow_pickle=False)
            gen_samples_1 = np.load("synthetic_pregen/synthetic_1.npy", allow_pickle=False)
            synth_data = np.concatenate((gen_samples_0, gen_samples_1), axis=0)



            results = np.zeros([len(models), 2])
            data_dim = synth_data.shape[1]

            for model_index in range(len(models)):
                model_name = models[model_index]

                results[model_index, 0], results[model_index, 1] = (
                    supervised_model_training(synth_data[:, :(data_dim - 1)],
                                              np.round(synth_data[:, (data_dim - 1)]),
                                              test_data[:, :(data_dim - 1)],
                                              np.round(test_data[:, (data_dim - 1)]),
                                              model_name))

            # Print the results for each iteration
            results = pd.DataFrame(np.round(results, 4),
                                   columns=['AUC-Synthetic Clean', 'APR-Synthetic Clean'])
            print(results)
            print('Averages:')
            print(results.mean(axis=0))

            results.mean(axis=0).to_csv(f"{folder_name}/results-pategan-{n + results_start}.csv", index=False)

            synth_data.tofile(f"{folder_name}/synth-data-pategan-{n + synth_data_start}.csv", sep=",")
final_testing()