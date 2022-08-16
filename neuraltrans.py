import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import requests
#import imageio

from PIL import Image


@tf.function
def load_image(binary_image):
    tensor_image = tf.image.decode_image(binary_image, channels=3)
    float_image = tf.image.convert_image_dtype(tensor_image, tf.float32)
    return float_image[tf.newaxis, :]


def download_image(url, fname):
    image = requests.get(url).content
    with open(fname, "wb") as local_file:
        local_file.write(image)
    return image


def display_images(source_image, style_image):

    plt.rcParams["figure.dpi"] = 100

    plt.imshow(np.squeeze(source_image))
    plt.title("source image")
    plt.show()

    plt.imshow(np.squeeze(style_image))
    plt.title("style image")
    plt.show()


if __name__ == "__main__":

    # Download images to local files and load

    binary_content_img = download_image(
        "https://imgur.com/kHZSqFd.jpeg",
        "src.jpg",
    )
    binary_style_img = download_image(
        "https://imgur.com/dPr5sKZ.jpeg",
        "style.jpg",
    )

    content_image = load_image(binary_content_img)
    style_image = load_image(binary_style_img)

    # Save combined image (?)

    img = Image.fromarray((np.squeeze(style_image) * 255).astype(np.uint8))
    img.save("generated_img.jpg")

    # Display downloaded images

    display_images(content_image, style_image)

    # Image stylization

    stylization_src = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    model = hub.load(stylization_src)
    stylized_image = model(
        tf.constant(content_image),
        tf.constant(style_image),
    )[0]