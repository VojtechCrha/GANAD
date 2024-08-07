import inspect
import os
import re

from post_gen_AD_pategan import pategan_main
import argparse


def find_highest_n(folder_path, pattern):

    # List all files in the specified folder
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return 0

    # Extract all n values from matching files
    n_values = []
    for file_name in files:
        match = pattern.match(file_name)
        if match:
            n_values.append(int(match.group(1)))

    # Return the highest n value or 0 if no matching files were found
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
            default='uci-epileptic',
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
            default=1000,
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
            help='PATE noise size',
            default=1.0,
            type=float)
        parser.add_argument(
            '--ad_budget',
            help='the budget for anomaly detection',
            default = 0.1,
            type=float)

        args = parser.parse_args()

        number_of_tests = 10

        #Create folder based on the parameters used
        folder_name = f"results/post_gen_eps={args.epsilon}_dataset={args.dataset}_iters={args.iterations}_k={args.k}_ad_budget={args.ad_budget}_delta={args.delta}_ns={args.n_s}_lambda={args.lamda}_noise_rate={args.noise_rate}_batchsize={args.batch_size}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)


        #Find the highest finished results file
        pattern_results = re.compile(r"results-pategan-(\d+)$")

        results_start = find_highest_n(folder_name, pattern_results)

        pattern_synth_data = re.compile(r"synthetic-data-pategan-(\d+)$")

        synth_data_start = find_highest_n(folder_name, pattern_synth_data)


        for n in range(1, number_of_tests+1):

            results, synth_data = pategan_main(args)

            results.mean(axis=0).to_csv(f"{folder_name}/results-pategan-{n+results_start}.csv", index=False)

            synth_data.tofile(f"{folder_name}/synth-data-pategan-{n+synth_data_start}.csv", sep=",")
final_testing()
