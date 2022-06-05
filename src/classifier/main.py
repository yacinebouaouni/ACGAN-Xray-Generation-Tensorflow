from model import model
import sys, getopt


def main(argv):

    try:
        opts, args = getopt.getopt(argv, ":d:m:")
    except getopt.GetoptError:
        print('model.py -d <data_path>')
        sys.exit(2)

    if (len(opts) == 0):
        print('model.py -d <data_path> -m <train/test>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('model.py -d <data_path> -m <train/test>')
            sys.exit()

        elif opt in ("-d", "--data"):
            data_path = arg

        elif opt in("-m", "--mode"):
            mode = arg

    print('Path to data ', data_path)
    print('########################### '+mode+' mode #############################')

    classifier = model(data_path)

    if mode == "train":
        classifier.train(epochs=100)

    if mode == "test":
        classifier.test()


if __name__ == '__main__':

    main(sys.argv[1:])
