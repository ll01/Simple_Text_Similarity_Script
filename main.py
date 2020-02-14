import sentencepiece as spm
import os
import csv
import argparse
import math
import time

import use
from common_functions import compare_items


def main():
    args = set_up_args()

    header, argos_items_data = readCsv(args.argos)
    print("argos data header: %s" % header)
    _, sainsbury_items_data = readCsv(args.sainsbury)
    comparison_engine = use.USE()

    item_start = 0

    batch_start = 0

    batch_count = math.ceil(len(argos_items_data) / args.batch_size)

    sainsbury_items_data = list(enumerate(sainsbury_items_data))[item_start:]

    for sainsbury_item_index, item in sainsbury_items_data:
        for batch in range(batch_start, batch_count):
            print("sainsbury\'s id {} item {} batch {} out of {}".format(
                item[0], sainsbury_item_index, batch + 1, batch_count))
            start_index = (batch * args.batch_size)
            end_index = min((start_index+args.batch_size),
                            len(argos_items_data))
            print("start_index: {}, end_index: {}".format(start_index, end_index))
            scores_with_ids = compare_items(
                item, argos_items_data[start_index:end_index],
                comparison_engine.embed)

            file_path = os.path.join(args.output, "{}.csv".format(item[0]))
            save_results(scores_with_ids, file_path)
            write_progress(sainsbury_item_index, batch)
        batch_start = 0


def set_up_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--argos', help='argos items csv', required=False,
        default="./argos.csv")
    parser.add_argument(
        '-s', '--sainsbury',
        help='sainsbury\'s items csv this is checked against the argos items'
        'to find maches', required=False, default="./sainsburys.csv")
    parser.add_argument('-o', '--output',
                        help='where the output directory', required=False,
                        default="./results/")
    parser.add_argument('-b', '--batch_size',
                        help='amount of argos items to compare in each batch',
                        required=False, type=int, default=10000)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def readCsv(file_path, delim=","):
    data = []
    with open(file_path,) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=delim)
        header = None
        for row in csvReader:
            data.append(row)
    header = data.pop(0)
    return header, data


def save_results(data, file_path):
    with open(file_path, "a+", newline='') as csvDataFile:
        for row in data:
            csvWriter = csv.writer(csvDataFile)
            csvWriter.writerow(row)


def write_progress(item, batch):
    with open("progress.txt", "w") as progress_file:
        progress_file.write("{}\n{}\n".format(item, batch))

if __name__ == "__main__":
    main()
