from src.models.model2 import ACGAN
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="path to data")
    parser.add_argument("-z", "--latentdim", type=int, required=True, help="latent dimension")
    parser.add_argument("-s", "--save", type=int, required=False, help="save model every s iterations")

    args = parser.parse_args()

    data_path = args.data
    z = args.latentdim
    s = args.save

    print('Path to data ', data_path)
    print('Latent Dimension ', z)
    print('save_every : ', s)

    acgan = ACGAN(data_path, z)
    acgan.train(epochs=800, save_every=s)


if __name__ == '__main__':
    main()
