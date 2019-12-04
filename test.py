from __future__ import absolute_import, division, print_function, unicode_literals
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from skimage.color import gray2rgb
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from skimage.transform import resize
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.applications import ResNet50
img_rows = 32
img_cols = 32
num_classes = 10

model_a = tf.keras.models.load_model("/Users/xtstc131/Documents/GitHub/CSI5138_Project/model/ckpt-DenseNet121_a.h5")
model_b = tf.keras.models.load_model("/Users/xtstc131/Documents/GitHub/CSI5138_Project/model/ckpt-DenseNet121_b.h5")

def preprocessData():
    # load and reshape the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    # # Input for others
    # else:
    #   x_train = x_train.reshape(60000, 784)
    #   x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def splitData(x_train, y_train, size):
    # Shuffle data randomly
    indexes = np.arange(size)
    np.random.shuffle(indexes)
    x = x_train[indexes]
    y = y_train[indexes]
    x_a = x[0:size // 2]
    x_b = x[size // 2:]
    y_a = y[0:size // 2]
    y_b = y[size // 2:]
    return x_a, x_b, y_a, y_b

x_train, y_train, x_test, y_test = preprocessData()
x_a, x_b, y_a, y_b = splitData(x_train, y_train, 50000)
# x_a = np.array([resize(img, (96,96)) for img in x_a[:,:,:,:]])
print(x_a.shape)
loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label, model):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad
# x_b = np.array([resize(img, (96,96)) for img in x_b[:,:,:,:]])
x_a = tf.convert_to_tensor(x_a[:5000], dtype=tf.float32)

image_probs = model_a.predict(x_a)
# print(image_probs)
perturbations = create_adversarial_pattern(x_a[:5000], image_probs, model=model_a)
def FSGM(img, perturbation, epsilons):
    adv_x = img + perturbations * epsilons
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x
model_name =  "DenseNet121"
shape_list =  [1, 32, 32, 3]
if model_name == "InceptionV3":
  shape_list = [1, 96, 96, 3]
adver = tf.reshape(adversarial_imgs[index], shape_list)
origin = tf.reshape(x_a[index], shape_list)
print("Adversarial Example result")
print(model_a.predict(adver).argmax())
print("Original Example result")
print(model_a.predict(origin).argmax())

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_id + "_adversarial_plot.png", format='png', dpi=300)
def plot_result(plot_num, model, adversarial_imgs, adversarial_lable, original_lable, plot_name):
    fig, axes = plt.subplots(plot_num, plot_num, figsize=(28, 28))
    imgs = tf.squeeze(adversarial_imgs)
    for x in range(plot_num):
        for y in range(plot_num):
            al = adversarial_lable[x * plot_num + y]
            ol = original_lable[x * plot_num + y]
            if al == ol:
                rect = patches.Rectangle((0, 0), 31, 31, linewidth=10, edgecolor='r', facecolor='none')
            else:
                rect = patches.Rectangle((0, 0), 31, 31, linewidth=10, edgecolor='g', facecolor='none')
            axes[x, y].imshow(imgs[x * plot_num + y])
            axes[x, y].add_patch(rect)
            axes[x, y].set_title(f'{al} ({ol})')

    save_fig(plot_name)

img_per_row = 5
# offset = np.random.randint(30000-img_per_row*img_per_row)
offset = 1
adv_imgs = adversarial_imgs[offset:offset + img_per_row * img_per_row]
adv_lable_a = a_lable[offset:offset + img_per_row * img_per_row]
org_lable_a = t_lable[offset:offset + img_per_row * img_per_row]
adv_lable_b = a_lable_b[offset:offset + img_per_row * img_per_row]
org_lable_b = t_lable_b[offset:offset + img_per_row * img_per_row]

plot_result(5, model_a, adv_imgs, adv_lable_a, org_lable_a, model_name + "_model_a")
plot_result(5, model_b, adv_imgs, adv_lable_b, org_lable_b, model_name + "_model_b")