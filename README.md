# CS4476 CV Project Repo

Project website can be found [here](https://jiachenren.github.io/cs4476-cv-project/).

### Where's the Paper?

Click [here](web/README.md) to see the rendered paper.

### How to contribute?

The different sections of the paper are broken into parts under [parts](parts). To work on a section, edit individual markdown files in there. When you are done, **make sure to execute the following script to recompile the final paper** and commit:

#### Compile

Make sure you are under the project directory, and have `python3`

```shell
chmod +x scripts/compile.py && scripts/compile.py
```

#### Images

To add images, put them under [images](images) directory. To link them from markdown, use the relative path. E.g. `[Alt Text](../images/<filename>)`

#### How to add a new section?

In [template.md](template.md), first add the line (order matters)

```markdown
[//]: # "section-name.md"
```

then, in [parts](parts), create the corresponding file `section-name.md`, edit the content of the section there.

## Database

We are using both self collected data the `eDBtheque` database.

### Data collection protocol

#### Self collected data

The self collected data contain several manga pages crawled from different websites. They are used purely for research purposes.

Currently, we have uploaded 2 chapters of 2 different comics from the romantized indonesian manga site [seltekomik](www.sektekomik.com)
to serve as our system's test data.

#### eDBtheque

The state-of-the-art manga database with ground truth pixel level labelling for panels and speech bubbles. It contains 100 pages in total,
and is used in most of the relevant researches pertaining to information retrieval (IR) from manga. 

If you are part of this project, contact Jiachen for the database login credentials. Otherwise request access from the owner [here](http://ebdtheque.univ-lr.fr/registration/)

## Project Dependencies & Installation Guide

### Install Tesseract OCR

This project makes use of google's [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition. In order
for the system to successfully, run, please install command line tool `tesseract` and add it to path. For macos, just run

```shell
brew install tesseract
```

For other systems, refer to [this guide](https://tesseract-ocr.github.io/tessdoc/Home.html). When you install
tesseract, you might encounter some non-fatal errors, just ignore them. You'll be fine as long as you have the final binary.

#### Install Tesseract python package

In the project directory (assuming that you have your venv created), run

```shell
pip3 install pytesseract
``` 

### Python dependencies

#### opencv-python

- Preprocess input image by applying threshold and de-noise to convert to binary
- `SIFT` related functionalities for extracting features from recognized text blocks
- Group contours in detected text blocks for character level segmentation
- Flood fill of speech bubbles using SIFT key points cluster centers as seeding coordinates
- Morphing of speech bubble binary mask to consume texts within

#### PIL

- Highlight text blocks
- Converting between color spaces, write back to disk
- Mask detected text blocks for iterative OCR

#### sklearn

-` MeanShift` clustering of `SIFT` descriptor matches in masked image to hypothesize 
new dialog bounding boxes
- `KMeans` clustering of pixels under flood-fill seed mask to extract dominant color