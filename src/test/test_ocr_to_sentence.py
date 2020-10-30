from src.ocr.sift_ocr import sift_ocr
from src.ocr.utils import preprocess
from PIL import Image
from src.ocr.TextBlockInfo import TextBlockInfoParser, TextBlockInfo
from typing import List, Dict
import imageio


def test_ocr_to_sentence():
    image_uri = '../data/indonesian/sektekomik.com/slime/4.png'
    np_image = imageio.imread(image_uri)
    np_image = preprocess(np_image)
    parser = TextBlockInfoParser()
    blocks_dict: Dict[int: List[TextBlockInfo]] = sift_ocr(Image.fromarray(np_image), parser)
    sentences = []
    for grp in range(1, len(blocks_dict)):
        blocks = blocks_dict[grp]
        # Todo: mean-shift cluster TextBlockInfo.bounds by its y coordinate first
        #  (with a window size of around 5 pixels)
        blocks.sort(key=lambda x: (x.bounds.origin()[1], x.bounds.origin()[0]))
        words = [x.text for x in blocks]
        separator = ' '
        sent = separator.join(words).lower()
        sentences.append(sent)
    for sent in sentences:
        print(sent)


test_ocr_to_sentence()
