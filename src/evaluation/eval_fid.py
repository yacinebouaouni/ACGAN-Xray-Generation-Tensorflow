from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.utils import calculate_fid
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--real", type=str, required=True, help="path to real dataset")
    parser.add_argument("-g", "--generated", type=str, required=True, help="path to generated dataset")
    parser.add_argument("-b", "--batch", type=int, required=True, help="batch size")
    args = parser.parse_args()

    data = args.real
    data2 = args.generated
    bs = args.batch

    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    train_datagen = ImageDataGenerator()
    train_generator_r = train_datagen.flow_from_directory(
        data+'real/',
        target_size=(299,299),
        batch_size=bs,
        class_mode='binary', shuffle=True)

    train_datagen = ImageDataGenerator()
    train_generator_f = train_datagen.flow_from_directory(
        data+'fake/',
        target_size=(299,299),
        batch_size=bs,
        class_mode='binary', shuffle=True)

    train_datagen = ImageDataGenerator()
    train_generator_f2 = train_datagen.flow_from_directory(
        data2,
        target_size=(299,299),
        batch_size=bs,
        class_mode='binary', shuffle=True)

    images1, _ = train_generator_r.next()
    images2, _ = train_generator_f.next()
    images3, _ = train_generator_f2.next()

    print('Prepared', images1.shape, images2.shape)
    # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    images3 = images3.astype('float32')

    # resize images
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    images3 = preprocess_input(images3)

    # fid between images1 and images1
    fid = calculate_fid(model, images1, images2)
    print('FID AC-GAN: %.3f' % fid)
    # fid between images1 and images2
    fid = calculate_fid(model, images1, images3)
    print('FID DC-GAN: %.3f' % fid)


if __name__ == "__main__":
    main()
