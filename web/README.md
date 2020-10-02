# Manga Text Extraction & Translation

### Contributors

Weiyao Tang, Zhaoran Ma, Jiachen Ren, Haoran Zhang, May Vy Le

### Abstract

[//]: # "abstract.md"
In this project, we aim to make the lives of manga translators easier by designing a system that takes in a manga image as input, extracts speech bubbles, then uses online translation API to translate the texts (originally in Japanese), finally producing a new image with translated speech bubbles as output.

// Todo: a few sentences about approach taken, results 

### Introduction

[//]: # "introduction.md"
Manga have been around for over centuries and hundreds of mangas are printed everyday in Japan. However, manga has recently gained a rise of popularity and one of the reasons is due to the internet; manga are now digitized into web content and there are now many hosting websites where anyone can upload their web manga and users can read for free. However, most manga are written in Japanese or Korean, and to share manga to non-Japanese/Korean readers, a translation to English is needed. Yet, any translation work is time consuming labor since there is no automatic method to translate the writing in manga into any other language. The goal of this project is that by using image captioning, a user can input an image with manga in the original language to the system, utilizing text detection and identification model to get the manga inside it and the desired output is the manga with the text translated to English.

### Approach

[//]: # "approach.md"
#### Steps

1. **Locating text in manga image**

   Using Mask R-CNN (an implementation can be found [here](https://github.com/cuppersd/MASKRCNN-TEXT-DETECTION)), 
   we can detect where the text is located in the image. However, this method of text extraction might prove to be too complex 
   to post-process as it also identifies and extracts texts from the drawing, 
   not only from speech bubbles. 
   
   An alternative approach is to first detect where structured text appear, remove them from the image, and then run edge
   detection to find speech bubbles. After texts are removed, it should be very easy to identify where the speech bubbles
   are using **canny edge detection**, after which we can limit our search window to these speech bubbles to identify japanese
   characters to translate.
   
   To identify Japanese characters (either Kanji or Hirigana), we can try the Mask R-CNN mentioned above, or alternatively
   we can try using hough transform to see how it performs. Hough transform is a valid approach here because each Japanese 
   character looks like a square (e.g. 私わ行きます) - we can downscale the image until they become circles, then use circle detection
   algorithm to locate them. 

2. **Token/Character Identification**

   After locating the speech bubbles, we will use OCR to identify the Japanese characters and prepare them for translation.
   
   In addition to OCR, we will identify the average width and average height of each character, the average horizontal spacing
   and vertical spacing between each character (these can be calculated by looking at the bounding rectangle for each character).
   This information is later used to generate and properly size translated text.
   
3. **Translation**
   
   Using [Google Translate API](https://cloud.google.com/translate/docs), we translate the japanese characters into target language.
   (In our case, target language is English) This part does not have too much to do with computer vision.

4. **Inserting Text Back to Image**

   After the translation is done, we will use the information about where the speech bubbles are located, their respective
   bounding rectangle, the size of the original text characters, and the original spacing to calculate the size, format, and position of
   the translated text in the target language to be in-painted. We will probably borrow from the script located [here](https://gist.github.com/destan/5540702) 

### Experiments and Results

[//]: # "experiments-and-results.md"
#### Data Set

To carry out this project, we need datasets of manga images to perform experiments on. However, despite our efforts, we couldn't locate any existing manga datasets in Japanese. What we do have at hand is a dataset of images already translated to English (for one of our member's NLP project). These images are collected using a web-crawler (written in dart) from this [website](https://mangasee123.com), and is purely used for research purposes. The crawler can be easily extended to collect raw manga resources from legitimate Japanese websites, and we intend to do that. 

However, if said resource is not attainable due to various reasons, we might change our objective to "translating from english to other languages." This is not too different from our current objective, since english texts in manga are in most cases **capitalized**, so we can easily employ hough transform for each letter or existing OCR methods to extract english texts, simplifying the problem statement.

It's unlikely that we'll train our own neural model for OCR, so we'll probably use pretrained models mentioned below.

#### List of Experiments

##### 1. Using Mask R-CNN to detect text boundaries

When documents are clearly laid out and have global structure (for example, a business letter), existing tools for OCR can perform quite well. In Manga, first, the document of interest occurs alongside some background objects . Second, the text within the document is highly unstructured and therefore it is beneficial to separately identify all the possible text blocks. Inspired by fully convolutional networks, we came up with the idea of modifying the model Mask R-CNN as an effective approach for text location, which is consisted of two steps. 

First, CNN is adopted to detect text blocks, from which character candidates are extracted. Then FPN is used to predict the corresponding segmentation masks. Last, segmentation mask is used to ﬁnd suitable rectangular bounding boxes for the text instances. The model generates bounding boxes and segmentation masks for each instance of an object in the image.

![Manga2](../images/manga2.png)

For the token/character identification, we plan to identify the Japanese characters in the regions where we detect some text in the first step. Here we want to apply Optical Character Recognition to text images and extract individual Japanese characters in sequence from those images. We already found some code that implements this model ([Japanese OCR](https://github.com/phamdinhthang/japanese_OCR) and [Hiragana Identifier](https://github.com/RakuTheSenpai/Hiragana-Identifier)). At the same time, we expect our output of this step to be some form of sentences so that we can translate the sentences in Japanese to English context later. Since text in manga is usually aligned vertically from top to bottom, we want to first identify separate text columns by detecting blank areas between them, and this can be possibly implemented using a simple gradient energy map. After getting the columns, we may also want to segment the columns on a character-level basis (including punctuations). Then we plan to apply the recognition model we mentioned above to classify each individual character and concatenate them into complete sentences.

![Manga2](../images/manga3.png)

##### 2. Using Hough Transform and Canny Edge Detection to find speech bubbles and text.

Another experiment that we will carry out as an alternative to Mask R-CNN is to use hough transform and canny edge detection to locate text in manga and extract speech bubble bounding rectangles. We will compare how each of these methods performed (or, in reality, feasible), and select one for our implementation.

In addition to simply extracting bounding box for speech bubbles, we will experiment with a novel idea of identifying exact boundaries of speech bubbles (which can help with in-painting boundaries later). It works like so - 

- First, individual text characters are detected, potentially using hough transform.
- Using **kmeans** or **mean-shift**, we detect text cluster centers, which are likely to be the center to speech bubbles (see images above for an example). 
- Then, hypothesize a bubble boundary by drawing a body that encompasses all text characters in each cluster. Then, using the "snake" algorithm, facilitated by the gradient of the speech bubble's boundary, we can assume the exact shape of the speech bubble.

#### Expected Outcomes

Ideally, our end system should be able to perform the following

1. Extract & locate text in manga
2. Locate speech bubbles in manga
3. Using OCR, convert text in image to ASCII
4. Translate the text, then using original position, size, and spacing information, in-paint the translated text into their corresponding speech bubbles in the new manga.

We realize that there are still uncertainties in our setup and also expectation, but we will try our best.





