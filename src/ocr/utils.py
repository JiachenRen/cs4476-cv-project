from PIL import ImageDraw, Image
from typing import List, Tuple
import numpy as np
import cv2
from src.ocr.TextBlockInfo import TextBlockInfo
from src.ocr.Rect import Rect
from sklearn.cluster import KMeans
from collections import Counter


def draw_blocks_on_image(img: Image.Image, blocks: List[TextBlockInfo], fill=(255, 0, 0), outline=None,
                         line_width=2, alpha=80) -> Image.Image:
    """
    Highlights detected text blocks on a copy of the image with supplied drawing parameters

    :param img: image to highlight text bocks on
    :param blocks: text blocks for the image
    :param line_width: line width of bounding box
    :param fill: bounding box fill (R, G, B)
    :param outline: bounding box outline (R, G, B)
    :param alpha: alpha of the drawing layer
    :return: a copy of img with text blocks highlighted
    """
    rgb_img = img.convert('RGBA')
    color_layer = Image.new('RGBA', rgb_img.size, fill)
    alpha_mask = Image.new('L', rgb_img.size, 0)
    draw = ImageDraw.Draw(alpha_mask)
    for block in blocks:
        draw.rectangle(block.bounds.corners(), outline=outline, fill=alpha, width=line_width)
    return Image.composite(color_layer, rgb_img, alpha_mask)


def find_dominant_colors(input_image: Image.Image, bounds: Rect, n: int, k: int) -> List[Tuple[int]]:
    """
    Finds dominant colors under specified area of an image
    :param input_image an RGB image
    :param n finds n most dominant colors
    :param k clusters to use
    :param bounds dominant color is found under this rectangle
    :return: list of n most dominant colors
    """
    colors = KMeans(n_clusters=k)
    # noinspection PyTypeChecker
    pixels_under_mask = np.array(input_image.crop(bounds.box())).reshape((-1, 3))
    colors.fit(pixels_under_mask)
    pixel_labels = colors.predict(pixels_under_mask)
    pixel_label_counter = Counter(pixel_labels)
    dominant_color_indices = [x[0] for x in pixel_label_counter.most_common(n)]
    return [tuple((round(x) for x in colors.cluster_centers_[idx])) for idx in dominant_color_indices]


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    De-noises the image and convert to gray space, then use a threshold to convert image to binary

    :param image: input image
    :return: an ndarray representing the image with thresholding and de-noise applied
    """
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # res, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return image


# SIFT group colors
sift_group_colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (100, 0, 255),
    (200, 255, 0),
    (200, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 100, 0),
    (0, 255, 100),
    (100, 255, 0),
    (0, 100, 255),
    (255, 0, 100),
    (255, 200, 0),
    (255, 0, 200),
    (0, 200, 255),
    (0, 255, 200)
]