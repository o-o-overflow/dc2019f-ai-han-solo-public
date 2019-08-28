#pylint:disable=unused-import

import ai_han_solo
import imageio
import keras
import numpy as np
import os

# dimensions of our images.
WORD_LENGTH=16
LETTER_WIDTH = 28
LETTER_HEIGHT = 28
COMPILE_OPTIONS = { 'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy'] }

def generator_attack(model_path, result_dir='output', tolerance=0.17, num_images=2**32):
    batch_size = 128
    sample_interval = 20
    samples = 10
    noise_shape = (100,)
    #img_shape = (LETTER_HEIGHT, LETTER_WIDTH*WORD_LENGTH, 1)

    # load and compile the discriminator
    discriminator = keras.models.load_model(model_path)
    discriminator.compile(**COMPILE_OPTIONS)

    # Build the generator
    generator_model = keras.models.Sequential()
    #generator_model.add(keras.layers.Dense(256, input_shape=noise_shape))
    #generator_model.add(keras.layers.LeakyReLU(alpha=0.2))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    #generator_model.add(keras.layers.Dense(512))
    #generator_model.add(keras.layers.LeakyReLU(alpha=0.2))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    #generator_model.add(keras.layers.Dense(1024))
    #generator_model.add(keras.layers.LeakyReLU(alpha=0.2))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    #generator_model.add(keras.layers.Dense(np.prod(img_shape), activation='tanh'))
    #generator_model.add(keras.layers.Reshape(img_shape))

    generator_model.add(keras.layers.Dense(128 * LETTER_WIDTH*WORD_LENGTH//4 * LETTER_HEIGHT//4, activation="relu", input_shape=noise_shape))
    generator_model.add(keras.layers.Reshape((LETTER_HEIGHT//4, LETTER_WIDTH*WORD_LENGTH//4, 128)))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.UpSampling2D())
    generator_model.add(keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("relu"))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.UpSampling2D())
    generator_model.add(keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("relu"))
    generator_model.add(keras.layers.Dense(128))
    generator_model.add(keras.layers.Activation("relu"))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("relu"))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("relu"))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("relu"))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("relu"))
    #generator_model.add(keras.layers.BatchNormalization(momentum=0.8))
    generator_model.add(keras.layers.Conv2D(1, kernel_size=3, padding="same"))
    generator_model.add(keras.layers.Activation("tanh"))


    generator_noise = keras.layers.Input(shape=noise_shape)
    generator_img = generator_model(generator_noise)
    generator = keras.models.Model(generator_noise, generator_img)

    # The generator takes noise as input and generates imgs
    z = keras.layers.Input(shape=noise_shape)
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity
    combined = keras.models.Model(z, valid)
    #combined.compile(**compile_options)
    combined.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # first, do some pre-training
    #if len(sys.argv) >= 4:
    #   pretrainer = Model(z, img)
    #   print "Doing some pre-training...", len(sys.argv)-3
    #   sample_images = np.stack([ np.expand_dims(imageio.imread(i) / 255., 2) for i in sys.argv[3:] ])
    #   sample_noise = np.random.normal(0, 1, (len(sample_images), 100))
    #   print "... loaded samples", len(sample_images)
    #   print sample_noise.shape
    #   print sample_images.shape
    #   pretrainer.compile(loss='binary_crossentropy', optimizer='rmsprop')
    #   pretrainer.fit(sample_noise, sample_images)
    #   print "... done!"

    num_classes = discriminator.output_shape[1]
    desired = 1/num_classes
    even_spread = [ desired ] * num_classes
    desired_result = np.array([ even_spread for _ in range(batch_size) ])
    written_images = 0

    for epoch in range(2**32):
        print("Epoch",epoch)
        train_noise = np.random.normal(0, 1, (batch_size, 100))

        # Train the generator
        g_loss = combined.train_on_batch(train_noise, desired_result)
        print("... metrics:", combined.metrics_names, g_loss)

        # Plot the progress
        #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            print("... checking sample images")
            gen_noise = np.random.normal(0, 1, (samples, 100))
            gen_imgs = generator.predict(gen_noise)
            predictions = discriminator.predict(gen_imgs)
            for img,pp in zip(gen_imgs, predictions):
                variances = [ abs(desired-p)/desired for p in pp ]
                big_variances = [ v for v in variances if v > tolerance ]
                print(pp)
                print("Variances: %s cumulative, %s max, %s big: %s" % (sum(variances), max(variances), len(big_variances), sorted(big_variances)))
                if len(big_variances) > 5:
                    continue

                print("... found even image %d!" % num_images)
                outpath = os.path.join(result_dir, 'out_v%d_%02d.png'%(len(big_variances), written_images))
                imageio.imsave(outpath, img.squeeze())
                written_images += 1
                if written_images >= num_images:
                    break


if __name__ == '__main__':
    import sys
    generator_attack(sys.argv[1])
