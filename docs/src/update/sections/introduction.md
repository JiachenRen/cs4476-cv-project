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