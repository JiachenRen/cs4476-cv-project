from src.ocr.sift_ocr import sift_ocr
from src.ocr.utils import preprocess
from PIL import Image
from src.ocr.TextBlockInfo import TextBlockInfoParser, TextBlockInfo
from typing import List, Dict
from sklearn.cluster import MeanShift
import numpy as np
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
        model = MeanShift(bandwidth=5)
        model.fit(np.array([x.bounds.y for x in blocks]).reshape(-1, 1))
        centers = model.cluster_centers_
        blocks.sort(key=lambda x: (centers[model.predict([[x.bounds.y]])[0]][0], x.bounds.x))
        words = [x.text for x in blocks]
        separator = ' '
        sent = separator.join(words).lower()
        sentences.append(sent)
    for sent in sentences:
        print(sent)


test_ocr_to_sentence()
