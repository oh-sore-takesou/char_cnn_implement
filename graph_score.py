import argparse
import json
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('draw graph of score')
    parser.add_argument('-d', dest='score_file', type=argparse.FileType('r'), required=True)
    return parser.parse_args()

def main(args):
    scores = args.score_file.readlines()
    X = range(len(scores))
    Y = scores
    plt.plot(X, Y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    args = get_args()
    main(args)
