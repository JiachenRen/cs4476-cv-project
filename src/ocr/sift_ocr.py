from PIL import Image, ImageDraw
from src.ocr.TextBlockInfo import parse_blocks_from_image
from src.ocr.Rect import Rect
from src.ocr.utils import draw_blocks_on_image
from typing import List, Tuple
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
    blocks = parse_blocks_from_image(image, 50, min_confidence=40)
    print(f'> Found {len(blocks)} blocks')
    text_line_blocks: List[np.ndarray] = []
    if p.exists(sift_ocr_path):
        sh.rmtree(sift_ocr_path)
    os.mkdir(sift_ocr_path)
    os.mkdir(p.join(sift_ocr_path, 'blocks'))
    os.mkdir(p.join(sift_ocr_path, 'block_keypoints'))
    for i in range(len(blocks)):
        block = blocks[i]
        image_block = image.crop(block.bounds.box())
        # noinspection PyTypeChecker
        text_line = np.array(image_block)
        text_line_blocks.append(text_line)
        print(f'> Block {i + 1} {block.confidence}%\t{block.text}')
        image_block.save(p.join(sift_ocr_path, 'blocks', f'block_{i + 1}.png'))

    # At this point, image_blocks holds small clips of letters,
    # blocks holds all of the detected blocks
    print('> Building vocabulary...')
    vocab_keypoints = []
    vocab_descriptors = []
    sift: cv.SIFT = cv.SIFT_create()
    for idx, text_line in enumerate(text_line_blocks):
        keypoints, descriptors = sift.detectAndCompute(text_line, None)
        if keypoints is None or descriptors is None:
            continue
        vocab_keypoints += list(keypoints)
        vocab_descriptors += list(descriptors)
        kp_image = text_line.copy()
        kp_image = cv.drawKeypoints(kp_image, keypoints, kp_image, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv.imwrite(p.join(sift_ocr_path, 'block_keypoints', f'{idx + 1}.png'), kp_image)

    # Mask the original image to hide where the features come from
    print('> Masking original image...')
    masked_image = image.copy()
    masked_image = draw_blocks_on_image(masked_image, blocks, fill=(255, 255, 255), alpha=255)
    masked_image.save(p.join(sift_ocr_path, 'masked_image.png'))
    # noinspection PyTypeChecker
    masked_image_arr = np.array(masked_image.convert('L'))
    print('> Extracting SIFT descriptors from input image...')
    img_keypoints, img_descriptors = sift.detectAndCompute(masked_image_arr, None)

    # Matching between vocab features and image features
    bf = cv.BFMatcher()
    print('> Matching between vocab and masked image...')
    matches: List[Tuple[cv.DMatch, cv.DMatch]] = bf.knnMatch(img_descriptors, np.array(vocab_descriptors), k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    print('> Painting good matches over masked image...')
    masked_image = masked_image.convert('RGBA')
    draw_on_masked = ImageDraw.Draw(masked_image, mode='RGBA')
    for match in good_matches:
        kp: cv.KeyPoint = img_keypoints[match.queryIdx]
        x, y = kp.pt
        w = h = 20
        rect = Rect(x - w / 2, y - h / 2, w, h)
        draw_on_masked.rectangle(rect.corners(), outline=(255, 0, 0), width=2)
    masked_image.save(p.join(sift_ocr_path, 'masked_image_text_matches.png'))


def detect_chars_in_image_block(image: np.ndarray) -> List[Rect]:
    """
    Detects character bounding boxes in a text block detected by Tesseract using opencv APIs

    First, image block is converted to gray scale, then a threshold is applied to convert to binary.
    OpenCV is used to detect separate, contours,
    then bounding boxes are draw around each contour to segment characters.

    :param image: an nd.array, must be unambiguously a line of readable characters, 1D binary image
    :return: bounding boxes, sorted from left to right, of all characters in the block
    """
    contours, hierarchy = \
        cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    char_bounds: List[Rect] = []
    for contour in contours:
        bound = Rect(*cv.boundingRect(contour))
        char_bounds.append(bound)

    # Sort char bounding boxes from left to right
    char_bounds.sort(key=lambda r: r.origin[0])

    return char_bounds
