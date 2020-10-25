from PIL import Image, ImageDraw
from src.ocr.TextBlockInfo import parseBlocksFromImage, TextBlockInfo
from typing import List
import pytesseract

test_img: Image.Image = Image.open('../data/indonesian/sektekomik.com/slime/15.png')


blocks: List[TextBlockInfo] = parseBlocksFromImage(test_img)
for block in blocks:
    print(block)

draw = ImageDraw.Draw(test_img, mode='RGBA')
for block in blocks:
    if block.height > 100:
        # Ignore bounding boxes whose height is greater than 100
        continue
    draw.rectangle([(block.left, block.top), (block.left + block.width, block.top + block.height)], fill=(255, 0, 0, 80))

# Save image with text bounding box to gen/image_text_boxes_unguided.png
test_img.save('../gen/image_text_boxes.png', 'PNG')
