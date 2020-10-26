from PIL import Image
from typing import List
import pytesseract
import re

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


class TextBlockInfoParser:

    def __init__(self, max_block_height=50, min_confidence=0, validation_regex=None):
        """
        :param max_block_height: max text block height, above which text blocks are discarded
        :param min_confidence: min confidence to include block
        :param validation_regex: regex to validate parsed text
        """
        self.validation_regex = re.compile(r'[a-zA-Z\?\.\,0-9\)\(]+') if validation_regex is None else validation_regex
        self.max_block_height = max_block_height
        self.min_confidence = min_confidence

    def parse_blocks_from_image(self, image: Image.Image) -> List[TextBlockInfo]:
        """
        :param image: image to detect text blocks from
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
                    and block.bounds.h <= self.max_block_height:
                # Discard empty blocks / blocks with low confidence
                if self.min_confidence is None or block.confidence >= self.min_confidence:
                    # Ensure block's detected text passes validation
                    if self.validation_regex.search(block.text):
                        blocks.append(block)
        return blocks
