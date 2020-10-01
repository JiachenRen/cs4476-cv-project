#### Data Set

For the text detection setp, we are going to utilize the image cropped from the websites. 

For the text recognition step, the data set used was taken from a Github repository, which contained 50 different characters and 20 different samples for each one. While sample size is small, a considerable level of certainty can be obtained utilizing a good learning model.


#### Experimental Steps

When documents are clearly laid out and have global structure (for example, a business letter), existing tools for OCR can perform quite well. In Manga, first, the document of interest occurs alongside some background objects . Second, the text within the document is highly unstructured and therefore it is beneficial to separately identify all the possible text blocks. Inspired by fully convolutional networks, we came up with the idea of modifying the model MaskRCNN as an effective approach for text location, which is consisted of two steps. 

First, CNN is adopted to detect text blocks, from which character candidates are extracted. Then FPN is used to predict the corresponding segmentation masks. Last, segmentation mask is used to ﬁnd suitable rectangular bounding boxes for the text instances. The model generates bounding boxes and segmentation masks for each instance of an object in the image.

![Manga2](../images/manga2.png)

For the token/character identification, we plan to identify the Japanese characters in the regions where we detect some text in the first step. Here we want to apply Optical Character Recognition to text images and extract individual Japanese characters in sequence from those images. We already found some code that implements this model ([Japanese OCR](https://github.com/phamdinhthang/japanese_OCR) and [Hiragana Identifier](https://github.com/RakuTheSenpai/Hiragana-Identifier)). At the same time, we expect our output of this step to be some form of sentences so that we can translate the sentences in Japanese to English context later. Since text in manga is usually aligned vertically from top to bottom, we want to first identify separate text columns by detecting blank areas between them, and this can be possibly implemented using a simple gradient energy map. After getting the columns, we may also want to segment the columns on a character-level basis (including punctuations). Then we plan to apply the recognition model we mentioned above to classify each individual character and concatenate them into complete sentences.

![Manga2](../images/manga3.png)

#### Expected Outcomes

We expect for each mango image input, after applying our model, we could gain the text content from it and translated into English. We are uncertain about the accuracy of the model that we are going to apply to train Japanese characters due to its structure. 


