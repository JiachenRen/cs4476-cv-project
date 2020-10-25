from PIL import Image
from src.ocr.TextBlockInfo import parse_blocks_from_image
from src.ocr.Rect import Rect
from typing import List
import os
import os.path as p
import shutil as sh
import numpy as np
import cv2


def sift_ocr(image: Image.Image, sift_ocr_path='../gen/sift_ocr'):
    """
    SIFT feature guided image OCR

    Again, this is our original idea. The algorithm works like this:
    First, use Tesseract OCR to extract initial text bounding boxes. Then,
    learn learn the SIFT features in these bounding boxes and use the SIFT
    features to hypothesize location of new bounding boxes.

    :param image: input image
    :param sift_ocr_path: path to store sift_ocr intermediaries
    :return:
    """
    blocks = parse_blocks_from_image(image, 50, min_confidence=80)
    print(f'> Found {len(blocks)} blocks')
    image_blocks: List[Image.Image] = []
    if p.exists(sift_ocr_path):
        sh.rmtree(sift_ocr_path)
    os.mkdir(sift_ocr_path)
    os.mkdir(p.join(sift_ocr_path, 'blocks'))
    for i in range(len(blocks)):
        block = blocks[i]
        print(f'> Block {i + 1} ({block.confidence})%')
        image_block = image.crop(block.bounds.box())
        image_blocks.append(image_block)
        # char_bounds = detect_chars_in_block(image_block)
        # print(f'> {len(char_bounds)} characters in block {i + 1}')
        # draw_on_block = ImageDraw.Draw(image_block, mode='RGBA')
        # for rect in char_bounds:
        #     print(f'\t{rect}')
        #     draw_on_block.rectangle(rect.corners(), fill=(255, 0, 0, 80))
        image_block.save(f'{sift_ocr_path}/blocks/block_{i + 1}.png')


def detect_chars_in_block(block: Image.Image) -> List[Rect]:
    """
    Detects character bounding boxes in a text block detected by Tesseract using opencv bindings

    First, image block is converted to gray scale, then a threshold is applied to convert the image to binary.
    OpenCV is used to detect separate, contours, then bounding boxes are draw around each contour to segment characters.

    :param block:
    :return:
    """
    # noinspection PyTypeChecker
    image = np.array(block)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # threshold

    contours, hierarchy = \
        cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    char_bounds: List[Rect] = []
    for contour in contours:
        bound = Rect(*cv2.boundingRect(contour))
        char_bounds.append(bound)

    # Sort char bounding boxes from left to right
    char_bounds.sort(key=lambda r: r.origin[0])

    return char_bounds
