from PIL import Image, ImageDraw
from typing import List, Tuple
from src.ocr.utils import draw_blocks_on_image
import numpy as np
import os
import shutil
import os.path as p

from src.ocr.TextBlockInfo import parse_blocks_from_image, TextBlockInfo


def iterative_ocr(image: Image.Image, max_iterations=5, iterative_ocr_path='../gen/iterative_ocr') \
        -> Tuple[Image.Image, Image.Image, List[TextBlockInfo]]:
    """
    The baseline Tesseract OCR isn't designed for detecting texts in manga.
    See ../gen/image_ocr_baseline.png.

    The modified OCR algorithm, iterative OCR, runs Tesseract OCR on the image multiple times;
    after each iteration, the detected texts are masked and then the masked image is fed into the next iteration.
    The algorithm stops after a specified number of iterations, or if no blocks are found after an iteration.
    See ../gen/iterative_ocr/ for results after each iteration.

    :param image: input image
    :param max_iterations: iterations to run
    :param iterative_ocr_path: path to store iterative OCR intermediaries
    :return: (im_masked, im_highlighted, text_blocks), where im_masked has
             all detected texts masked, im_highlighted has all detected texts highlighted
    """
    # noinspection PyTypeChecker
    masked_image = Image.fromarray(np.array(image))
    # noinspection PyTypeChecker
    highlighted_image = Image.fromarray(np.array(image))
    blocks: List[TextBlockInfo] = []
    if p.exists(iterative_ocr_path):
        shutil.rmtree(iterative_ocr_path)
    os.mkdir(iterative_ocr_path)
    for i in range(max_iterations):
        print(f'> Iteration {i + 1}')
        new_blocks = parse_blocks_from_image(masked_image)
        if len(new_blocks) == 0:
            print('> No text detected, stopping.')
            break
        # Mask recognized text blocks
        masked_image = draw_blocks_on_image(masked_image, new_blocks, fill=(255, 255, 255), alpha=255)
        # Highlight detected text blocks in red
        highlighted_image = draw_blocks_on_image(highlighted_image, new_blocks)
        print(f'\tfound {len(new_blocks)} new text blocks')
        highlighted_image.save(p.join(iterative_ocr_path, f'iteration_{i + 1}_highlighted.png'))
        masked_image.save(p.join(iterative_ocr_path, f'iteration_{i + 1}_masked.png'))
        blocks += new_blocks
    return masked_image, highlighted_image, blocks




