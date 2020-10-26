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

#### PIL

- Highlight text blocks
- Converting between color spaces, write back to disk
- Mask detected text blocks for iterative OCR

#### sklearn

-` MeanShift` clustering of `SIFT` descriptor matches in masked image to hypothesize 
new dialog bounding boxes