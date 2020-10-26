from PIL import Image
from src.ocr.TextBlockInfo import parse_blocks_from_image, TextBlockInfo
from src.ocr.iterative_ocr import iterative_ocr
from src.ocr.utils import draw_blocks_on_image, preprocess
from typing import List, Tuple
import numpy as np
import imageio

from src.ocr.sift_ocr import sift_ocr

image_uri = '../data/indonesian/sektekomik.com/slime/7.png'


def load_test_image() -> Tuple[np.ndarray, Image.Image]:
    np_image = imageio.imread(image_uri)
    np_image = preprocess(np_image)
    return np_image, Image.fromarray(np_image)


def test_baseline_ocr():
    # Test baseline OCR (using google's Tesseract OCR)
    test_img: Image.Image = Image.open(image_uri)
    blocks: List[TextBlockInfo] = parse_blocks_from_image(test_img)
    print(f'Baseline OCR (found {len(blocks)} blocks) ------------------------------')
    for block in blocks:
        print(block)

    highlighted = draw_blocks_on_image(test_img, blocks)

    # Save image with text bounding box to gen/image_ocr_baseline.png
    highlighted.save('../gen/ocr_baseline.png')


def test_preprocessed_ocr():
    """
    Same as base line OCR, but applies a preprocessing procedure first (denoise, threshold)
    """
    np_image, pil_image = load_test_image()

    blocks: List[TextBlockInfo] = parse_blocks_from_image(pil_image)
    print(f'Preprocessed OCR (found {len(blocks)} blocks) ------------------------------')
    for block in blocks:
        print(block)

    highlighted = draw_blocks_on_image(pil_image, blocks)

    # Save image with text bounding box to gen/image_ocr_baseline.png
    highlighted.save('../gen/ocr_preprocessed.png')


def test_iterative_ocr():
    # Test iterative OCR, with preprocessing applied
    np_image, pil_image = load_test_image()
    masked_image, highlighted_image, blocks = iterative_ocr(pil_image, max_iterations=5)
    print(f'Iterative OCR (found {len(blocks)} blocks) ------------------------------')
    for block in blocks:
        print(block)


def test_sift_ocr():
    np_image, pil_image = load_test_image()
    sift_ocr(pil_image)


if __name__ == '__main__':
    test_baseline_ocr()
    test_preprocessed_ocr()
    test_iterative_ocr()
    test_sift_ocr()
