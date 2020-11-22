from typing import Dict, List, Tuple, Callable
from PIL import Image, ImageDraw, ImageFont
from src.ocr.Rect import Rect
from src.ocr.utils import find_dominant_colors
from src.ocr.TextBlockInfo import TextBlockInfo
from sklearn.cluster import MeanShift
import numpy as np


class PaintingParams:
    def __init__(self, bounds, line_height, lines):
        """
        :param bounds: bounds for the bubble
        :param line_height: height of each line
        :param lines: how many lines to fit in the bubble
        """
        self.bounds: Rect = bounds
        self.line_height: float = line_height
        self.lines: int = lines

    def line_spacing(self):
        spacing = (self.bounds.h - self.line_height * self.lines) / self.lines
        return spacing if spacing > 0 else 5

    def font_size(self):
        return int(self.line_height)

    def split_to_lines(self, text: str, text_length: Callable[[str], float]) -> List[str]:
        """
        Splits text into multiple lines with word-level warp around based on painting params
        :param text: input text
        :param text_length: fn to calculate length of text
        :return: list of lines
        """
        lines = []
        words = text.split(' ')
        i = 0
        c_width = 0
        buffer = []
        while i < len(words):
            word = words[i]
            if c_width > self.bounds.w:
                if len(buffer) > 1:
                    i -= 1
                    buffer = buffer[:-1]
                lines.append(' '.join(buffer))
                c_width = 0
                buffer = []
                continue
            c_width += text_length(word + ' ')
            buffer.append(word)
            i += 1
        if len(buffer) > 0:
            if c_width > self.bounds.w and len(buffer) > 1:
                lines.append(' '.join(buffer[:-1]))
                lines.append(buffer[-1])
            else:
                lines.append(' '.join(buffer))
        return lines


def compute_painting_params(blocks_dict: Dict[int, List[TextBlockInfo]], lines: List[int]) -> List[PaintingParams]:
    """
    Compute in-painting parameters from blocks dict returned by SIFT OCR
    :param blocks_dict blocks dict returned by calling sift_ocr
    :param lines how many lines are in each block
    :return: in-painting bounds and text sizes for each group
    """
    painting_params = []
    # Start from group 1, since group 0 is every group combined
    for grp in range(1, len(blocks_dict)):
        blocks: List[TextBlockInfo] = blocks_dict[grp]
        min_x, min_y, max_x, max_y = 1000000, 1000000, 0, 0
        line_height = 0
        for block in blocks:
            bounds: Rect = block.bounds
            line_height += bounds.h
            if bounds.x < min_x:
                min_x = bounds.x
            if bounds.diagonal()[0] > max_x:
                max_x = bounds.diagonal()[0]
            if bounds.y < min_y:
                min_y = bounds.y
            if bounds.diagonal()[1] > max_y:
                max_y = bounds.diagonal()[1]
        line_height /= len(blocks)
        bounds = Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        painting_params.append(PaintingParams(bounds, line_height, lines[grp - 1]))
    return painting_params


def extract_texts(blocks_dict: Dict[int, List[TextBlockInfo]]) -> Tuple[List[str], List[int]]:
    """
    Reconstructs texts from each group of text blocks; computes lines in each group
    :param blocks_dict: result returned by calling sift_ocr
    :return: tuple of (texts, lines) for each group
    """
    texts = []
    # How many lines are in each group
    lines = []
    # Start from group 1, since group 0 is every group combined
    for grp in range(1, len(blocks_dict)):
        blocks = blocks_dict[grp]
        # Mean-shift cluster text blocks to normalize rows
        model = MeanShift(bandwidth=5)
        model.fit(np.array([x.bounds.y for x in blocks]).reshape(-1, 1))
        centers = model.cluster_centers_
        lines.append(len(centers))
        # Sort by x, then by y, then by x to reconstruct texts in original order
        blocks.sort(key=lambda x: (centers[model.predict([[x.bounds.y]])[0]][0], x.bounds.x))
        words = [x.text for x in blocks]
        separator = ' '
        sent = separator.join(words).lower()
        texts.append(sent)
    return texts, lines


def inpaint(image: Image.Image, translations: List[str], params: List[PaintingParams], font_path: str) -> Image.Image:
    """
    Paints translated text over original image using specified painting params
    :param image: input image
    :param font_path: path to .ttf font file to use
    :param translations: list of translated texts for each group
    :param params: list of painting params for each group
    :return: image with translated text painted over
    """
    rgb_image = image.convert('RGB')
    draw = ImageDraw.Draw(rgb_image)
    for i in range(len(translations)):
        param = params[i]
        translation = translations[i].upper()
        bounds: Rect = param.bounds
        dominant_colors = find_dominant_colors(rgb_image, bounds, 1, 5)
        background_color = dominant_colors[0]
        text_color = (0, 0, 0)
        # Mask original text
        draw.rectangle(bounds.corners(), fill=background_color)
        # Load font
        font_size = param.font_size()
        font = ImageFont.truetype(font_path, size=font_size)
        lines = param.split_to_lines(translation, lambda w: draw.textlength(w, font))
        line_spacing = param.line_spacing()
        x, y = bounds.origin()
        for line in lines:
            draw.text((x, y), line, fill=text_color, font=font)
            y += line_spacing + param.line_height
    return rgb_image
