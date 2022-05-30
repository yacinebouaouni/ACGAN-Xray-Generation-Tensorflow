from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def generate(model_path, covid_n, noncovid_n, latent_dim=128):

    model = load_model(model_path)
    Path("CovidCXR").mkdir(parents=True, exist_ok=True)
    Path("NonCovidCXR").mkdir(parents=True, exist_ok=True)

    for i in range(covid_n):
        np.random.seed(i)
        noise = np.random.normal(0, 0.02, (1, latent_dim))
        sampled_labels = np.array(0)
        X = model.predict([sampled_labels.reshape((-1, 1)), noise], verbose=0)
        X = (X * 127.5 + 127.5).astype(np.uint8)
        print(noise[0])
        image = Image.fromarray(X[0, :, :, 0], "L")
        image.save("CovidCXR/" + str(i + 1) + ".jpeg")

    for i in range(noncovid_n):
        np.random.seed(i)
        noise = np.random.normal(0, 0.02, (1, latent_dim))
        sampled_labels = np.array(1)
        X = model.predict([sampled_labels.reshape((-1, 1)), noise], verbose=0)
        X = (X * 127.5 + 127.5).astype(np.uint8)

        image = Image.fromarray(X[0, :, :, 0], "L")
        image.save("NonCovidCXR/" + str(i + 1) + ".jpeg")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model")
    parser.add_argument("-c", "--covid", type=int, required=True, help="Number of covid images to generate")
    parser.add_argument("-n", "--noncovid", type=int, required=True, help="Number of non-covid images to generate")
    args = parser.parse_args()

    model_to_load = args.model
    covid_nbr = args.covid
    noncovid_nbr = args.noncovid

    print('Covid ', covid_nbr)
    print('Non-Covid ', noncovid_nbr)
    print('Model ', model_to_load)

    generate(model_to_load, covid_nbr, noncovid_nbr)


if __name__ == "__main__":
    main()
