import argparse
from learning.nn_learner import learn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in --input", action="store", dest="input_path", required=True,
                        help="path to the input data file")
    parser.add_argument("-sw --stop-words", action="store", dest="stopwords_path", required=True,
                        help="path to the stop words file")
    parser.add_argument("-w --window", action="store", type=int, dest="window", default=5,
                        help="window of context words")
    parser.add_argument("-b --batch-size", action="store", type=int, dest="batch_size", default=128,
                        help="size of mini-batches")
    parser.add_argument("-e --eta", action="store", type=float, dest="eta", default=1e-3,
                        help="learning rate - eta")
    parser.add_argument("--epochs", action="store", type=int, dest="epochs", default=10,
                        help="iterations over neural network - epochs")

    args = parser.parse_args()
    learn(args)

