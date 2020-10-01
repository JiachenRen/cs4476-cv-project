### Approach

#### Steps

1. Text Searching

   Using Mask R-CNN, we can detect where the text is located in the image and extract these characters for next step.

2. Token/Character Identification

   Applying OCR to text tokens and extract individual Japanese characters in sequence from those images

3. Inserting Text Back to Image

   Using machine translation API to translate the text content extracted to English