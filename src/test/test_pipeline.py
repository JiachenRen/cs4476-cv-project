from src.inpainting import extract_texts, compute_painting_params, inpaint
from src.ocr.sift_ocr import sift_ocr
from src.ocr.utils import preprocess
from PIL import Image
from src.ocr.TextBlockInfo import TextBlockInfoParser, TextBlockInfo
from typing import List, Dict
import os.path as p
import imageio

from src.translation.google_translator import GoogleTranslator, ClientType


def test_pipeline(image_uri: str, working_dir='../gen'):
    np_image = imageio.imread(image_uri)
    np_image = preprocess(np_image)
    parser = TextBlockInfoParser()
    print('-------------------- SIFT OCR -------------------')
    blocks_dict: Dict[int: List[TextBlockInfo]] = sift_ocr(Image.fromarray(np_image), parser)

    # Convert blocks to texts and lines.
    print('\n------------ Reconstruct Sentences -------------')
    texts, lines = extract_texts(blocks_dict)
    # Combine all sentences to be translated into one document, which allows us to only
    # call the API once to translate everything.
    combined_texts = ''
    for sent in texts:
        combined_texts += sent + '\n'
    # Each line in combined_texts is texts extracted from a single bubble.
    print(combined_texts)
    # Save extracted sentences
    with open(p.join(working_dir, 'sentences.txt'), 'w') as file:
        file.write(combined_texts)

    translator = GoogleTranslator(client_type=ClientType.siteGT)
    translation = translator.translate(combined_texts)
    print('\n----------------- Translation ------------------')
    print(translation.translated)
    # Save translated text
    with open(p.join(working_dir, 'translated.txt'), 'w') as file:
        file.write(translation.translated)
    translations = translation.translated.split('\n')

    # In-painting
    print('\n----------------- In-painting ------------------')
    print('> Computing painting params...')
    params = compute_painting_params(blocks_dict, lines)
    print('> In-painting...')
    final_image = inpaint(Image.open(image_uri), translations, params, font_path='../assets/fonts/Comic Sans MS.ttf')
    # Save final result
    path = p.join(working_dir, 'final_result.png')
    print(f'> Saving result to {path}...')
    final_image.save(path)
    print('> Done.')


if __name__ == '__main__':
    test_pipeline('../data/indonesian/sektekomik.com/slime/5.png')
