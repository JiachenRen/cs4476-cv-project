from PIL import Image, ImageDraw
from src.ocr.Rect import Rect
from src.ocr.iterative_ocr import iterative_ocr
from src.ocr.TextBlockInfo import TextBlockInfo, TextBlockInfoParser
from src.ocr.utils import draw_blocks_on_image, find_dominant_colors, sift_group_colors
from typing import List, Tuple, Dict
from sklearn.cluster import MeanShift, KMeans
from collections import Counter
import random
import os
import os.path as p
import shutil as sh
import numpy as np
import cv2 as cv


def sift_ocr(image: Image.Image, parser: TextBlockInfoParser, sift_ocr_path='../gen/sift_ocr', morph_rect_size=40,
             mean_shift_bandwidth=80, min_cluster_label_count=2, sift_match_threshold=0.7,
             flood_mask_size=50, flood_tolerance=(5, 5, 5)) -> Dict[int, List[TextBlockInfo]]:
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
    ✓ Using opencv, morph the bubbles to cover the texts within
    ✓ Use the bubble as a binary mask to mask irrelevant parts of input image
    ✓ Run boundary detection on bubbles to extract boundary
    ✓ Extract bounding box from boundary (opencv)
    ✓ Use these new bounding boxes to crop the masked input image,
      then run Tesseract OCR over each to extract more text.
    ✓ Results from different bounding boxes are put into different groups

    :param image: input image
    :param parser: text block info parser to use
    :param min_cluster_label_count: min number of points labelled for a cluster to keep it
    :param sift_match_threshold: threshold to keep sift matches, between 0-1, (smaller value = stricter match)
    :param flood_mask_size: size of the mask to use at cluster centers to facilitate flooding
    :param morph_rect_size: size of the structuring element used to fill characters in text balloons
    :param flood_tolerance: max allowed flood error in (R, G, B) when flooding speech bubbles
    :param sift_ocr_path: path to store sift_ocr intermediaries
    :param mean_shift_bandwidth: bandwidth for mean shift clustering of matched keypoints from input image
    :return: a dictionary with keys to denote group number, and values are extracted text blocks in the group
             dict[0] contains all blocks detected by iterative OCR
    """
    # Clear files from last run
    if p.exists(sift_ocr_path):
        sh.rmtree(sift_ocr_path)
    os.mkdir(sift_ocr_path)
    os.mkdir(p.join(sift_ocr_path, 'blocks'))
    os.mkdir(p.join(sift_ocr_path, 'block_keypoints'))
    os.mkdir(p.join(sift_ocr_path, 'masked'))
    os.mkdir(p.join(sift_ocr_path, 'masks'))

    # Save parameters
    params = f"""
Parser
    - max_block_height: {parser.max_block_height}
    - min_confidence: {parser.min_confidence}
    - validation_regex: {parser.validation_regex.pattern}

Sift OCR
    - morph_rect_size: {morph_rect_size}
    - mean_shift_bandwidth: {mean_shift_bandwidth}
    - min_cluster_label_count: {min_cluster_label_count}
    - sift_match_threshold: {sift_match_threshold}
    - flood_mask_size: {flood_mask_size}
    - flood_tolerance: {flood_tolerance}
"""
    with open(p.join(sift_ocr_path, 'parameters.txt'), 'w') as file:
        file.write(params)

    # Run iterative OCR, and put results into grouped_blocks[0]
    grouped_blocks: Dict[int, List[TextBlockInfo]] = {}
    _, _, blocks = iterative_ocr(image, parser)
    grouped_blocks[0] = blocks
    print(f'> Iterative OCR found {len(blocks)} blocks')

    # Crop text blocks found using iterative OCR and save them to blocks dir
    text_line_blocks: List[np.ndarray] = []
    for i in range(len(blocks)):
        block = blocks[i]
        image_block = image.crop(block.bounds.box())
        # noinspection PyTypeChecker
        text_line = np.array(image_block)
        text_line_blocks.append(text_line)
        print(f'> Block {i + 1} {block.confidence}%\t{block.text}')
        image_block.save(p.join(sift_ocr_path, 'blocks', f'block_{i + 1}.png'))

    # At this point, text_line_blocks holds small clips of letters,
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

    # Match vocab features with image features
    bf = cv.BFMatcher()
    print('> Matching')
    matches: List[Tuple[cv.DMatch, cv.DMatch]] = bf.knnMatch(img_descriptors, np.array(vocab_descriptors), k=2)

    # Apply ratio test to select good matches
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
    mean_shift = MeanShift(cluster_all=False, bandwidth=mean_shift_bandwidth)
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
        w = h = flood_mask_size
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
    struct_element = cv.getStructuringElement(cv.MORPH_RECT, (morph_rect_size, morph_rect_size))
    groups = 1
    for idx, c in enumerate(centers):
        print(f'> Hypothesizing bounding box from center {idx + 1}, {c}')
        flood_image = image.convert('RGB')
        draw = ImageDraw.Draw(flood_image)
        w = h = flood_mask_size
        x, y = tuple(c)
        mask_rect = Rect(x - w / 2, y - h / 2, w, h)

        # Separate pixels under mask to 2 colors, and choose the dominant one to apply under mask
        dominant_color = find_dominant_colors(flood_image, mask_rect, 1, 5)[0]
        draw.rectangle(mask_rect.corners(), fill=dominant_color)
        # noinspection PyTypeChecker
        flood_image = np.array(flood_image)
        flood_mask = np.zeros((flood_image.shape[0] + 2, flood_image.shape[1] + 2), np.uint8)

        # Flood a copy of input image, starting from center, using the dominant color
        cv.floodFill(
            flood_image,
            flood_mask,
            tuple(c),
            (0, 255, 0),
            loDiff=flood_tolerance,
            upDiff=flood_tolerance,
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

        # Run detection, save mask and masked image (with new detections drawn over)
        cv.imwrite(p.join(sift_ocr_path, 'masks', f'{idx + 1}.png'), flood_mask)
        new_blocks: List[TextBlockInfo] = parser.parse_blocks_from_image(masked_image)
        masked_image = draw_blocks_on_image(masked_image, new_blocks)
        masked_image.save(p.join(sift_ocr_path, 'masked', f'{idx + 1}.png'))

        # Translate block coordinates from masked image to input image
        for block in new_blocks:
            block.bounds.translate(flood_mask_bounds.origin())
        print(f'\tfound {len(new_blocks)} new blocks')
        if len(new_blocks) > 0:
            grouped_blocks[groups] = new_blocks
            groups += 1

    # Save results from iterative ocr to gen/image_ocr_baseline.png
    random.shuffle(sift_group_colors)
    iter_image = image.copy()
    iterative_results = draw_blocks_on_image(iter_image, grouped_blocks[0])
    iterative_results.save(f'{sift_ocr_path}/ocr_result_iterative.png')

    # Save grouped results
    grp_results_img = image.copy()
    for grp in range(1, len(grouped_blocks)):
        grp_results_img = draw_blocks_on_image(grp_results_img, grouped_blocks[grp], fill=sift_group_colors[grp])
    grp_results_img.save(f'{sift_ocr_path}/ocr_result_grouped.png')

    return grouped_blocks


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
