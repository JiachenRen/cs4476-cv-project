from PIL import Image, ImageDraw
from src.ocr.Rect import Rect
from src.ocr.iterative_ocr import iterative_ocr
from src.ocr.TextBlockInfo import TextBlockInfo, TextBlockInfoParser
from typing import List, Tuple
from sklearn.cluster import MeanShift, KMeans
from collections import Counter
import os
import os.path as p
import shutil as sh
import numpy as np
import cv2 as cv


def sift_ocr(image: Image.Image, parser: TextBlockInfoParser, sift_ocr_path='../gen/sift_ocr',
             min_cluster_label_count=2, sift_match_threshold=0.7, mask_size=50, max_flood_err=(5, 5, 5)) \
        -> List[TextBlockInfo]:
    """
    To overcome blind spots of Tesseract OCR, we developed SIFT feature guided image OCR.

    The algorithm works like this:
    ✓ Use Tesseract OCR to extract initial text bounding boxes.
    ✓ Run Iterative OCR until no more text can be extracted (See iterative_ocr.py)
    ✓ Learn SIFT descriptors from extracted bounding boxes to build the vocabulary.
    ✓ Extract descriptors from input image
    ✓ Find good matches between vocab descriptors and input image descriptors,
      these are likely places where Tesseract OCR failed to recognize text.
    ✓ Use MeanShift to cluster keypoints of matched descriptors to hypothesize text box centers
    ✓ Mask image at centers
    ✓ Flood fill (opencv) using centers as starting points
    - Run boundary detection on flooded areas to extract boundary
    - Extract bounding box from boundary (opencv)
    - Use these new bounding boxes to crop input image,
      then run Tesseract OCR over each to extract more text.

    :param image: input image
    :param parser: text block info parser to use
    :param min_cluster_label_count: min number of points labelled for a cluster to keep it
    :param sift_match_threshold: threshold to keep sift matches, between 0-1, (smaller value = stricter match)
    :param mask_size: size of the mask to use at cluster centers
    :param max_flood_err: max allowed flood error in (R, G, B) when flooding speech bubbles
    :param sift_ocr_path: path to store sift_ocr intermediaries
    :return:
    """
    _, _, blocks = iterative_ocr(image, parser)
    print(f'> Iterative OCR found {len(blocks)} blocks')
    text_line_blocks: List[np.ndarray] = []
    if p.exists(sift_ocr_path):
        sh.rmtree(sift_ocr_path)
    os.mkdir(sift_ocr_path)
    os.mkdir(p.join(sift_ocr_path, 'blocks'))
    os.mkdir(p.join(sift_ocr_path, 'block_keypoints'))
    os.mkdir(p.join(sift_ocr_path, 'masked'))
    os.mkdir(p.join(sift_ocr_path, 'masks'))
    for i in range(len(blocks)):
        block = blocks[i]
        image_block = image.crop(block.bounds.box())
        # noinspection PyTypeChecker
        text_line = np.array(image_block)
        text_line_blocks.append(text_line)
        print(f'> Block {i + 1} {block.confidence}%\t{block.text}')
        image_block.save(p.join(sift_ocr_path, 'blocks', f'block_{i + 1}.png'))

    # At this point, image_blocks holds small clips of letters,
    # blocks holds all of the detected blocks
    print('> Building vocabulary')
    vocab_keypoints = []
    vocab_descriptors = []
    sift: cv.SIFT = cv.SIFT_create()
    for idx, text_line in enumerate(text_line_blocks):
        keypoints, descriptors = sift.detectAndCompute(text_line, None)
        if keypoints is None or descriptors is None:
            continue
        vocab_keypoints += list(keypoints)
        vocab_descriptors += list(descriptors)
        kp_image = text_line.copy()
        kp_image = cv.drawKeypoints(kp_image, keypoints, kp_image, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv.imwrite(p.join(sift_ocr_path, 'block_keypoints', f'{idx + 1}.png'), kp_image)

    print('> Extracting SIFT descriptors')
    # noinspection PyTypeChecker
    img_keypoints, img_descriptors = sift.detectAndCompute(np.array(image.convert('L')), None)

    # Matching between vocab features and image features
    bf = cv.BFMatcher()
    print('> Matching')
    matches: List[Tuple[cv.DMatch, cv.DMatch]] = bf.knnMatch(img_descriptors, np.array(vocab_descriptors), k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < sift_match_threshold * n.distance:
            good_matches.append(m)

    sift_image = image.convert('RGBA')
    draw_on_sift = ImageDraw.Draw(sift_image, mode='RGBA')

    # Point of interests - where potential dialog bounding boxes should be
    poi = np.zeros((len(good_matches), 2))
    for idx, match in enumerate(good_matches):
        kp: cv.KeyPoint = img_keypoints[match.queryIdx]
        x, y = kp.pt
        poi[idx, :] = np.array([x, y])

    print('> Finding match cluster centers')
    mean_shift = MeanShift(cluster_all=False, bandwidth=80)
    mean_shift.fit(poi)
    centers = mean_shift.cluster_centers_
    poi_labels = mean_shift.predict(poi)

    # Discard cluster centers with label count lower than min_cluster_label_count
    label_counter = Counter(poi_labels)

    # Draw points by cluster red if not discarded, dark gray otherwise
    for i in range(len(poi)):
        x, y = poi[i]
        label = poi_labels[i]
        color = (255, 0, 0) if label_counter[label] >= min_cluster_label_count and label != -1 else (100, 100, 100)
        w = h = 20
        rect = Rect(x - w / 2, y - h / 2, w, h)
        draw_on_sift.rectangle(rect.corners(), outline=color, width=2)

    # Draw cluster centers green if not discarded, gray otherwise
    valid_centers = []
    for i in range(len(centers)):
        x, y = centers[i]
        count = label_counter[i]
        w = h = mask_size
        rect = Rect(x - w / 2, y - h / 2, w, h)
        if count >= min_cluster_label_count:
            valid_centers.append(centers[i])
        color = (0, 255, 0) if count >= min_cluster_label_count else (200, 200, 200)
        draw_on_sift.rectangle(rect.corners(), outline=color, width=3)
    centers = np.array(valid_centers, dtype='uint')
    sift_image.save(p.join(sift_ocr_path, 'matches_from_sift.png'))

    print(f'> Found {len(centers)} valid centers')
    # noinspection PyTypeChecker
    input_image_arr = np.array(image)
    # Structuring element to close text gaps in speech bubbles
    struct_element = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
    for idx, c in enumerate(centers):
        print(f'> Hypothesizing bounding box from center {idx + 1}, {c}')
        flood_image = image.convert('RGB')
        draw = ImageDraw.Draw(flood_image)
        w = h = mask_size
        x, y = tuple(c)
        mask_rect = Rect(x - w / 2, y - h / 2, w, h)

        # Separate pixels under mask to 2 colors, and choose the dominant one to apply under mask
        colors = KMeans(n_clusters=5)
        # noinspection PyTypeChecker
        pixels_under_mask = np.array(flood_image.crop(mask_rect.box())).reshape((-1, 3))
        colors.fit(pixels_under_mask)
        pixel_labels = colors.predict(pixels_under_mask)
        pixel_label_counter = Counter(pixel_labels)
        dominant_color_idx = pixel_label_counter.most_common(1)[0][0]
        dominant_color = (round(x) for x in colors.cluster_centers_[dominant_color_idx])
        draw.rectangle(mask_rect.corners(), fill=tuple(dominant_color))
        # noinspection PyTypeChecker
        flood_image = np.array(flood_image)
        flood_mask = np.zeros((flood_image.shape[0] + 2, flood_image.shape[1] + 2), np.uint8)

        # Flood a copy of input image, starting from center, using the dominant color
        cv.floodFill(
            flood_image,
            flood_mask,
            tuple(c),
            (0, 255, 0),
            loDiff=max_flood_err,
            upDiff=max_flood_err,
            flags=4 | cv.FLOODFILL_MASK_ONLY | (255 << 8))

        # Processing the flooded area a bit to make it a perfect speech bubble
        flood_mask = cv.morphologyEx(flood_mask, cv.MORPH_CLOSE, struct_element)
        flood_mask = flood_mask[2:, 2:]
        masked_image = cv.bitwise_and(input_image_arr, input_image_arr, mask=flood_mask)

        # Paint masked area white
        white_image = np.full_like(masked_image, 255)
        masked_image += cv.bitwise_and(white_image, white_image, mask=np.bitwise_not(flood_mask))

        # Find bounding rect of mask and use it to crop masked image
        flood_mask_contours, _ = cv.findContours(flood_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        flood_mask_bounds = Rect(*cv.boundingRect(flood_mask_contours[0]))
        masked_image = Image.fromarray(masked_image).crop(flood_mask_bounds.box())

        # Save mask and masked image
        cv.imwrite(p.join(sift_ocr_path, 'masks', f'{idx + 1}.png'), flood_mask)
        masked_image.save(p.join(sift_ocr_path, 'masked', f'{idx + 1}.png'))

        new_blocks: List[TextBlockInfo] = parser.parse_blocks_from_image(masked_image)
        for block in new_blocks:
            block.bounds.translate(flood_mask_bounds.origin())
        print(f'\tfound {len(new_blocks)} new blocks')
        blocks += new_blocks
    return blocks


def detect_chars_in_image_block(image: np.ndarray) -> List[Rect]:
    """
    Detects character bounding boxes in a text block detected by Tesseract using opencv APIs

    First, image block is converted to gray scale, then a threshold is applied to convert to binary.
    OpenCV is used to detect separate, contours,
    then bounding boxes are draw around each contour to segment characters.

    :param image: an nd.array, must be unambiguously a line of readable characters, 1D binary image
    :return: bounding boxes, sorted from left to right, of all characters in the block
    """
    contours, hierarchy = \
        cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    char_bounds: List[Rect] = []
    for contour in contours:
        bound = Rect(*cv.boundingRect(contour))
        char_bounds.append(bound)

    # Sort char bounding boxes from left to right
    char_bounds.sort(key=lambda r: r.origin[0])

    return char_bounds
