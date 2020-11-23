#### Dependencies

**pytesseract**

Our system builds on top of existing state-of-the-art OCR technology, [Google's Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
Tesseract OCR, currently maintained by google, has a long and robust history and is indisputably the go-to
open source OCR engine. Tesseract OCR is capable of recognizing many different languages, but it is designed to recognize
text in a structured document (think of a scanned novel, paper rendered with latex, etc.), not for recognizing text in an
image that consists mostly of art. However, as we have tested, it is indeed capable of extracting some texts along with their
bounding box from a manga page, which is enough for our purpose. Note that Tesseract has a python API wrapper in the form of a package,
[pytesseract](https://pypi.org/project/pytesseract/). The API is very basic as it only invokes the commandline API of Tesseract,
which is installed separately, and returns a CSV of extracted texts and their respective bounding boxes. To adapt it for our project,
we made a higher level API wrapper for `pytesseract`, which can be found [here](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/ocr/TextBlockInfo.py).

**opencv-python**

We use existing functions of [opencv-python](https://pypi.org/project/opencv-python/) for the following tasks:
- Preprocess input image by applying threshold and de-noise
- `SIFT` related functionalities for extracting features from recognized text blocks
- Flood fill of speech bubbles using SIFT key points cluster centers as seeding coordinates
- Morphing (dilation and erosion) of speech bubble binary mask to erase text contours

**sklearn**

We use several clustering algorithms from [sklearn](https://scikit-learn.org/stable/) including [mean-shift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) and [kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) in several places. We will explain the usage in detail in the [SIFT-OCR Algorithm](#sift-ocr-algorithm) section.

#### SIFT-OCR Algorithm

As previously mentioned in the [Dependencies](#dependencies) section, Tesseract performs well on structured documents, not on mangas (we will demonstrate this as baseline results in the Experiments & Results section). To adapt Tesseract OCR so it can perform well on manga pages, we designed the SIFT-OCR algorithm. If you wish to look at the python source code, you can find it [here](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/ocr/sift_ocr.py) - we believe that our code is extremely well documented, so it is strongly encouraged to read it. Here are the details of this algorithm, broken down into steps:

1. Use Tesseract OCR on the input image to extract initial text bounding boxes. At this point, each bounding box contains a line of text/characters.
2. Run Iterative OCR until no more text can be extracted (See [iterative_ocr.py](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/ocr/iterative_ocr.py)).
3. Learn SIFT descriptors from extracted bounding boxes to build the vocabulary. Essentially, we are learning language and font dependent characteristics of the text.
4. Extract SIFT descriptors from the input image.
5. Find good matches between vocabulary descriptors (descriptors extracted from lines of text) and input image descriptors - these are likely places where Tesseract OCR failed to recognize text.
6. Use MeanShift clustering on keypoint coordinates of matched descriptors to hypothesize speech bubble centers, apply thresholding on number of instances in each center to throw away some "bad" centers.
7. Extract pixels from arbitrarily sized rectangles centered at each cluster center from step 6, then use KMeans clustering on the pixels to find the dominant color - this is the background color of the speech bubble.
8. Flood fill speech bubbles. The cluster centers from step 6 are used as seeding coordinate, and the background color from step 7 is used as seeding color for each speech bubble respectively.
9. At this point, we have obtained binary masks shaped like speech bubbles. However, since flood fill is applied using the background color as seeding color, the texts in the bubble are not part of the mask. To morph the mask to consume the texts within, apply dilation and erosion filters using opencv.
10. Use the bubble as a binary mask to mask irrelevant parts of input image.
11. Run boundary detection on speech bubble masks, then calculate their bounding box using the detected boundary  (again, using opencv).
12. For each of the detected binary speech bubble mask and their bounding box, first apply the binary mask so only the text within the speech bubble remains, then crop the image using the bounding box so the text becomes centered.
13. Run Tesseract OCR again on each of the cropped, text-only images to detect and group found text blocks. The output of SIFT-OCR is a dictionary of `[GroupIndex: List[TextBlockInfo]]` that groups detected text blocks by the index of speech bubbles that the text belongs to.

The algorithm is based on the following assumptions:

1. Most speech bubbles have a closed, solid boundary (otherwise flood fill won't work). Although during experimentation, we found that the algorithm still achieves respectable results for non-closed-boundary speech bubbles.
2. Speech bubbles have mostly uniform background color (which is true). This is also a prerequisite for flood fill.
3. Tesseract OCR can extract some initial text bounding boxes (otherwise we can't build SIFT vocabulary)

Intuitively, since we know that SIFT is good at finding places that are similar to the vocabulary (in our case texts), SIFT-OCR
combines the advantages of SIFT and Tesseract OCR. Essentially, we first use Tesseract OCR to establish some ground truth of what text should
look like in this particular manga page. Then, using SIFT, we find all places where text might be in the page (what Tesseract OCR is not made to do).
Then, using this info, we can "de-noise" the input and guide Tesseract to work on speech bubbles only. Since text in speech bubbles appear as structured documents,
Tesseract achieves far better results than the baseline.

#### Translation & In-painting

To reconstruct sentences from each group of text blocks, first a `MeanShift` clustering with window size of `5` is done on the y coordinates of the corresponding bounding boxes to detect lines, then the text blocks are sorted first by their y coordinate (rounded to the nearest cluster center) and then by their x coordinate. 

Next, we use Google Translate to translate the obtained sentences in the source language. There's not much to talk about here except that we combine everything extracted from a manga page into a single document and use that as the query to the Google Translate API, which minimizes asynchronous network operations. For our implementation (ported from dart language), refer to [google_translator.py](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/translation/google_translator.py) and [google_token_generator.py](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/translation/google_token_generator.py).

Finally, for each grouped text blocks, we calculate various parameters related to painting the translated text back onto the original image. These parameters include font size, line spacing, color, and warp-around parameters. Drawing text over image is done using `PIL` package. For more details, refer to [inpainting.py](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/inpainting.py). Detailed documentations are provided there.

The pipeline runner can be found at [test_pipeline.py](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/test/test_pipeline.py)

In the next section, we will present qualitative results from each stage of our proposed pipeline and compare final results with baseline results.