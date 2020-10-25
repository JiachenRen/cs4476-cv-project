from PIL import Image
from typing import List
import pytesseract

from src.ocr.Rect import Rect


class TextBlockInfo:

    def __init__(self, level: int, page_num: int, block_num: int, par_num: int, line_num: int, word_num: int, left: int,
                 top: int, width: int, height: int, conf: int, text: str):
        self.level = int(level)
        self.page_num = int(page_num)
        self.block_num = int(block_num)
        self.par_num = int(par_num)
        self.line_num = int(line_num)
        self.word_num = int(word_num)
        self.bounds = Rect(int(left), int(top), int(width), int(height))
        self.confidence = int(conf)
        self.text = text.strip()  # Remove white spaces

    def __str__(self):
        return f'''>>> TextBlockInfo
      par.line.word: par. {self.par_num}, line {self.line_num}, word {self.word_num}
         confidence: {self.confidence}%
             bounds: {self.bounds}
               text: "{self.text}"
'''


def parse_blocks_from_image(image: Image.Image, max_block_height=100, min_confidence: int = None) \
        -> List[TextBlockInfo]:
    """
    :param max_block_height: max text block height, above which text blocks are discarded
    :param image: image to detect text blocks from
    :param min_confidence: min confidence to include block
    :return: list of text blocks extracted from image
    """
    data = pytesseract.image_to_data(image).split('\n')
    # Remove data header
    data.pop(0)
    blocks: List[TextBlockInfo] = []
    for line in data:
        if len(line) == 0:
            continue
        elements = line.split('\t')
        block = TextBlockInfo(*elements)
        if block.confidence != -1 and len(block.text) != 0 \
                and block.bounds.height() <= max_block_height:
            # Discard empty blocks / blocks with low confidence
            if min_confidence is None or block.confidence >= min_confidence:
                blocks.append(block)
    return blocks
