# Manga Text Detection, Extraction & Translation

### Contributors

Weiyao Tang, Zhaoran Ma, Jiachen Ren, Haoran Zhang, May Vy Le

### Abstract

[//]: # "abstract.md"
[//]: # "
One or two sentences on the motivation behind the problem you are solving. 
One or two sentences describing the approach you took. 
One or two sentences on the main result you obtained.
Teaser figure that conveys the main idea behind the project or the main application being addressed.
"

### Introduction

[//]: # "introduction.md"
[//]: # "
Motivation behind the problem you are solving, 
what applications it has, 
any brief background on the particular domain you are working in (if not regular RBG photographs), etc. 
If you are using a new way to solve an existing problem, 
briefly mention and describe the existing approaches and tell us how your approach is new.
"
Manga, or comics, is a form of art that combines story telling and art. In recent years, this form of media has garnered attention
across language, culture and geological boundaries. This is largely due to an increasing number of mangas being made available online.
To meet the demand of the manga consumers all over the world, many translators are working hard to translate manga from their original
language (also called "RAW") into the language of their targeted audience. 
Our proposed system aims to automate the workflow of translating manga pages. However, before diving into the details of our method,
 it is necessary to first understand the structure of a manga page and the workflow of translators.

A typical manga page is broken down into several components - panels, speech bubbles, lines of text, and art.
To construct a manga page, several panels are laid out in the page, each containing a piece of art. 
Panels are not necessarily rectangular, and does not necessarily have boundaries - it is a very flexible form of layout.
Then, speech bubbles are overlaid on top of the panels (or some times clipped by the panels), each of which containing
lines of text detailing the story.

As for the workflow of a translator, it includes locating speech bubbles, manually transcribing text from the source language,
then translating the text from source language to target language (which requires expertise in both), then finally, formatting
the translated text so it can fit back into the original speech bubble.

Unsurprisingly, many of the steps above are tedious and can be automated using modern deep learning approaches. However,
despite the fact that manga is full of interesting information that is suitable for supervised ML tasks such as object recognition,
image segmentation, machine translation, etc., there is a lack of
dataset and ground truth labelling for supervised machine learning tasks. This is due to copyright protection and many other
caveats that come with intellectual property. As a result, most of the research done in this area either use self-collected datasets that
are undisclosed or the [eDBtheque](http://ebdtheque.univ-lr.fr) dataset, a dataset of 100 pages with pixel-level ground truth labelling for panels.
The results from these researches that employ custom designed deep neural nets either focus on specific tasks such as speech bubble localization
of mangas in a specific language and genre (**Todo: references here**), or text bounding box extraction for mangas in a specific language (**Todo: references here**).
The point is, none of these produce an end-to-end system with high level of performance.

Our approach introduces several innovations in designing an end-to-end pipeline for automated manga translation. In tackling this
challenging problem, we combined traditional machine learning and computer vision algorithms with deep learning methods to achieve
results far better than the baseline approach. Specifically, we come up with the **SIFT guided OCR** algorithm to extract text from
manga speech bubbles. We show that our algorithm performs far better than the baseline in the industry with qualitative results.

**Todo: Teaser Figure**

### Approach

[//]: # "approach.md"
[//]: # "
Describe very clearly and systematically your approach to solve the problem. 
Tell us exactly what existing implementations you used to build your system. 
Tell us what obstacles you faced and how you addressed them. 
Justify any design choices or judgment calls you made in your approach.
"

#### Dependencies

##### Tesseract OCR
Our system builds on top of existing state-of-the-art OCR technology, [Google's Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
Tesseract OCR, currently maintained by google, has a long and robust history and is indisputably the go-to
open source OCR engine. Tesseract OCR is capable of recognizing many different languages, but it is designed to recognize
text in a structured document (think of a scanned novel, paper rendered with latex, etc.), not for recognizing text in an
image that consists mostly of art. However, as we have tested, it is indeed capable of extracting some texts along with their
bounding box from a manga page, which is enough for our purpose. Note that Tesseract has a python API wrapper in the form of a package,
[pytesseract](https://pypi.org/project/pytesseract/). The API is very basic as it only invokes the commandline API of Tesseract,
which is installed separately, and returns a CSV of extracted texts and their respective bounding boxes. To adapt it for our project,
we made a higher level API wrapper for `pytesseract`, which can be found [here](https://github.com/JiachenRen/cs4476-cv-project/blob/master/src/ocr/TextBlockInfo.py).

##### Open CV 


### Experiments and Results

[//]: # "experiments_and_results.md"
[//]: # "
 Provide details about the experimental set up (number of images/videos, number of datasets you experimented with, train/test split if you used machine learning algorithms, etc.). 
 Describe the evaluation metrics you used to evaluate how well your approach is working. 
 Include clear figures and tables, as well as illustrative qualitative examples if appropriate. 
 Be sure to include obvious baselines to see if your approach is doing better than a naive approach (e.g. for classification accuracy, how well would a classifier do that made random decisions?). 
 Also discuss any parameters of your algorithms, and tell us how you set the values of those parameters. 
 You can also show us how the performance varies as you change those parameter values. 
 Be sure to discuss any trends you see in your results, and explain why these trends make sense. 
 Are the results as expected? Why?
"

#### Qualitative Results

[//]: # "experiments_and_results.md"

### Conclusion

[//]: # "conclusion.md"
[//]: # "
Conclusion would likely make the same points as the abstract. Discuss any future ideas you have to make your approach better.
"

### References

[//]: # "references.md"
[//]: # "
List out all the references you have used for your work.
"

