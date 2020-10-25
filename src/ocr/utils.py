from PIL import ImageDraw, Image
from typing import List
import numpy as np
import cv2
from src.ocr.TextBlockInfo import TextBlockInfo


def highlight_blocks_on_image(img: Image.Image, blocks: List[TextBlockInfo], fill=(255, 0, 0), outline=None,
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


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    De-noises the image and convert to gray space, then use a threshold to convert image to binary

    :param image: input image
    :return: an ndarray representing the image with thresholding and de-noise applied
    """
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return image
