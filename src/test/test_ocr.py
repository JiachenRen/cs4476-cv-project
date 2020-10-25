from PIL import Image, ImageDraw
from src.ocr.TextBlockInfo import parse_blocks_from_image, TextBlockInfo
from src.ocr.iterative_ocr import iterative_ocr
from typing import List

max_block_height = 100
imageUri = '../data/indonesian/sektekomik.com/demon_king/1.png'
test_img: Image.Image = Image.open(imageUri)

# Test baseline OCR (using google's Tesseract OCR)
blocks: List[TextBlockInfo] = parse_blocks_from_image(test_img)
print(f'Baseline OCR (found {len(blocks)} blocks) ------------------------------')
for block in blocks:
    print(block)

test_img_highlighted = test_img.copy()
draw = ImageDraw.Draw(test_img_highlighted, mode='RGBA')
for block in blocks:
    if block.bounds.height() > max_block_height:
        # Ignore bounding boxes whose height is greater than 100
        continue
    draw.rectangle(block.bounds.corners(), fill=(255, 0, 0, 80))

# Save image with text bounding box to gen/image_ocr_baseline.png
test_img_highlighted.save('../gen/image_ocr_baseline.png', 'PNG')

# Test iterative OCR
masked_image, highlighted_image, blocks = iterative_ocr(imageUri, max_iterations=3, max_block_height=max_block_height)
print(f'Iterative OCR (found {len(blocks)} blocks) ------------------------------')
for block in blocks:
    print(block)
