from src.ocr.sift_ocr import sift_ocr
from src.ocr.utils import preprocess
from PIL import Image
from src.ocr.TextBlockInfo import TextBlockInfoParser, TextBlockInfo
from typing import List, Dict
from sklearn.cluster import MeanShift
import numpy as np
import imageio

from src.translation.google_translator import GoogleTranslator, ClientType


def parse_bounds_from_blocks_dict(blocks_dict: Dict[int: List[TextBlockInfo]]):
    """
    Parses in-painting bounds and text sizes from blocks dict returned by SIFT OCR
    :return: in-painting bounds and text sizes for each group
    """
    # Todo: complete this
    painting_bounds = {}
    # for grp in blocks_dict.keys():


def test_pipeline():
    image_uri = '../data/indonesian/sektekomik.com/slime/4.png'
    np_image = imageio.imread(image_uri)
    np_image = preprocess(np_image)
    parser = TextBlockInfoParser()
    blocks_dict: Dict[int: List[TextBlockInfo]] = sift_ocr(Image.fromarray(np_image), parser)

    # Blocks to sentences
    sentences = []
    for grp in range(1, len(blocks_dict)):
        blocks = blocks_dict[grp]
        model = MeanShift(bandwidth=5)
        model.fit(np.array([x.bounds.y for x in blocks]).reshape(-1, 1))
        centers = model.cluster_centers_
        blocks.sort(key=lambda x: (centers[model.predict([[x.bounds.y]])[0]][0], x.bounds.x))
        words = [x.text for x in blocks]
        separator = ' '
        sent = separator.join(words).lower()
        sentences.append(sent)

    # Translation
    combined_sentences = ''
    for sent in sentences:
        combined_sentences += sent + '\n'
    translator = GoogleTranslator(client_type=ClientType.siteGT)
    translation = translator.translate(combined_sentences)
    print(translation.translated)


test_pipeline()
