from PIL import Image, ImageDraw
from typing import List, Tuple

from src.ocr.TextBlockInfo import parse_blocks_from_image, TextBlockInfo


def iterative_ocr(imageUri: str, max_iterations=5, max_block_height=100) \
        -> Tuple[Image.Image, Image.Image, List[TextBlockInfo]]:
    """
    The baseline Tesseract OCR isn't designed for detecting texts in manga.
    See ../gen/image_ocr_baseline.png.

    The modified OCR algorithm, iterative OCR, runs Tesseract OCR on the image multiple times;
    after each iteration, the detected texts are masked and then the masked image is fed into the next iteration.
    The algorithm stops after a specified number of iterations, or if no blocks are found after an iteration.
    See ../gen/iterative_ocr/ for results after each iteration.

    :param imageUri: URI of the image
    :param max_iterations: iterations to run
    :param max_block_height: max block height for text blocks
    :return: (im_masked, im_highlighted, text_blocks), where im_masked has
             all detected texts masked, im_highlighted has all detected texts highlighted
    """
    masked_image = Image.open(imageUri)
    highlighted_image = Image.open(imageUri)
    draw_on_masked = ImageDraw.Draw(masked_image)
    draw_on_highlighted = ImageDraw.Draw(highlighted_image, mode='RGBA')
    blocks: List[TextBlockInfo] = []
    for i in range(max_iterations):
        print(f'> Iteration {i + 1}')
        new_blocks = parse_blocks_from_image(masked_image)
        if len(new_blocks) == 0:
            print('> No text detected, stopping.')
            break
        for block in new_blocks:
            if block.bounds.height() < max_block_height:
                # Mask recognized text blocks
                draw_on_masked.rectangle(block.bounds.corners(), fill=(255, 255, 255))
                draw_on_highlighted.rectangle(block.bounds.corners(), fill=(255, 0, 0, 80))
        print(f'\tfound {len(new_blocks)} new text blocks')
        highlighted_image.save(f'../gen/iterative_ocr/iteration_{i + 1}_highlighted.png')
        masked_image.save(f'../gen/iterative_ocr/iteration_{i + 1}_masked.png')
        blocks += new_blocks
    return masked_image, highlighted_image, blocks




