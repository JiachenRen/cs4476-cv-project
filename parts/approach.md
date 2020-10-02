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