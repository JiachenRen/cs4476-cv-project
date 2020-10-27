# CS4476 CV Project Repo

Project website can be found [here](https://jiachenren.github.io/cs4476-cv-project/).

## Paper

The source for the three required deliverables (proposal, 2 updates) are broken down into 3 parts under [docs/src](docs/src).
The files are organized like so under docs

```
.
├── compile.py
├── proposal.md
└── src
    └── proposal
        ├── images
        │   ├── mainfig.png
        │   └── ... other images
        ├── index.md
        └── sections
            ├── abstract.md
            └── ... other sections
    ├── update
        ├── ... same structure as proposal
    ├── ... other parts
```

### Deliverables

Click [here](docs/proposal.md) to see the proposal.

Click [here](docs/update.md) to see the first midterm update.

Click [here](docs/final.md) to see the (second) final update.

### How to contribute?
There are three versions, `proposal`, `update`, and `final`, and the source files for these are located under their respective
directory under [docs/src]. 

Each section of each version have their respective file under `sections` directory for each update. 
To work on a section, edit individual markdown files in there. **Edit these, not the rendered version**.
 
When you are done, **execute the following script to recompile** and commit:

#### Compile

Make sure you are under the `docs` directory, and have `python3`. 
The compiler will generate the final paper for each version (`proposal.md`, `update.md`, `final.md`) and put them under [docs](docs)

```shell
chmod +x compile.py && compile.py
```

#### Images

To add images, put them under `images` directory for the correct version. Yes, each version has its own `images` directory. 
To link them from markdown, use the relative path. E.g. `[Alt Text](../images/<filename>)`

#### How to add a new section?

In `index.md` for the version you are working on, first add the line (order matters).

```markdown
[//]: # "<section_name>.md"
```

Then, in `sections.md` under `sections` directory for the version, create the corresponding file `section-name.md`, edit the content of the section there.

## Database

We are using both self collected data the `eDBtheque` database.

### Data collection protocol

#### Self collected data

The self collected data contain several manga pages crawled from different websites. They are used purely for research purposes.

Currently, we have uploaded 2 chapters of 2 different comics from the romantized indonesian manga site [sektekomik](www.sektekomik.com)
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