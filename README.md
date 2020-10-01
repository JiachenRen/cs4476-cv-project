## Manga Text Extraction & Translation

### Abstract

One or two sentences on the motivation behind the problem you are solving. One or two sentences describing the approach you took. One or two sentences on the main result you obtained.

### Introduction

Manga have been around for over centuries and hundreds of mangas are printed everyday in Japan. However, manga has recently gained a rise of popularity and one of the reasons is due to the internet; manga are now digitized into web content and there are now many hosting websites where anyone can upload their web manga and users can read for free. However, most manga are written in Japanese or Korean, and to share manga to non-Japanese/Korean readers, a translation to English is needed. Yet, any translation work is time consuming labor since there is no automatic method to translate the writing in manga into any other language. The goal of this project is that by using image captioning, a user can input an image with manga in the original language to the system, utilizing text detection and identification model to get the manga inside it and the desired output is the manga with the text translated to English.

### Approach

#### Steps

1. Text Searching

   Using Mask R-CNN, we can detect where the text is located in the image and extract these characters for next step.

2. Token/Character Identification

   Applying OCR to text tokens and extract individual Japanese characters in sequence from those images

3. Inserting Text Back to Image

   Using machine translation API to translate the text content extracted to English
   
Todo: generate