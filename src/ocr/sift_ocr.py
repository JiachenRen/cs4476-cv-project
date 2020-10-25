from PIL import Image
from src.ocr.TextBlockInfo import parse_blocks_from_image
from src.ocr.Rect import Rect
from typing import List
import os
import os.path as p
import shutil as sh
import numpy as np
import cv2 as cv


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
    text_line_blocks: List[np.ndarray] = []
    if p.exists(sift_ocr_path):
        sh.rmtree(sift_ocr_path)
    os.mkdir(sift_ocr_path)
    os.mkdir(p.join(sift_ocr_path, 'blocks'))
    os.mkdir(p.join(sift_ocr_path, 'block_keypoints'))
    for i in range(len(blocks)):
        block = blocks[i]
        print(f'> Block {i + 1} ({block.confidence})%')
        image_block = image.crop(block.bounds.box())
        # noinspection PyTypeChecker
        text_line = np.array(image_block)
        text_line_blocks.append(text_line)
        # char_bounds = detect_chars_in_block(image_block_arr)
        image_block.save(f'{sift_ocr_path}/blocks/block_{i + 1}.png')

    # At this point, image_blocks holds small clips of letters,
    # blocks holds all of the detected blocks
    for idx, text_line in enumerate(text_line_blocks):
        sift: cv.SIFT = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(text_line, None)
        kp_image = text_line.copy()
        kp_image = cv.drawKeypoints(kp_image, keypoints, kp_image, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv.imwrite(p.join(sift_ocr_path, 'block_keypoints', f'{idx + 1}.png'), kp_image)


def detect_chars_in_image_block(image: np.ndarray) -> List[Rect]:
    """
    Detects character bounding boxes in a text block detected by Tesseract using opencv APIs

    First, image block is converted to gray scale, then a threshold is applied to convert to binary.
    OpenCV is used to detect separate, contours,
    then bounding boxes are draw around each contour to segment characters.

    :param image: an nd.array, must be unambiguously a line of readable characters
    :return: bounding boxes, sorted from left to right, of all characters in the block
    """
    # noinspection PyTypeChecker
    image = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply a threshold to convert image to binary
    res, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)

    contours, hierarchy = \
        cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    char_bounds: List[Rect] = []
    for contour in contours:
        bound = Rect(*cv.boundingRect(contour))
        char_bounds.append(bound)

    # Sort char bounding boxes from left to right
    char_bounds.sort(key=lambda r: r.origin[0])

    return char_bounds
