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

#### Data

Initially, we have planned to use a database with ground truth labelling to train a deep learning model to detect speech bubbles and structured text. However, the acquisition of the only known dataset for such a task, **eDBtheque**, requires prior authorization by the researchers at France. Due to this, there was a delay in getting the database - by that time we've already developed a non-deep-learning, traditional CV and ML oriented pipeline to accomplish our objective. We do, however, plan to use the database for the final
stage of our project if time permits.

In order to run experiments, test hypothesis, and assess how well our algorithm performs compared to the baseline, we have collected our own dataset. Our dataset consists of two random chapters of two mangas crawled from the indonesian manga website, [sektekomik.com](https://sektekomik.com). Here are some random pages from our dataset:

<table>
	<tr>
		<td>
			<img src="../images/sample_pages/3.png"></img>
		</td>
		<td>
			<img src="../images/sample_pages/11.png"></img>
		</td>
		<td>
			<img src="../images/sample_pages/12.png"></img>
		</td>
	</tr>
</table>
Our dataset contains manga pages with romantized Indonesian text. All the experimental results below are obtained from these pages. However, do note that our system is scalable to other languages - Tesseract is multilingual and SIFT learns from Tesseract results. 



#### SIFT-OCR Pipeline

In this section, we will show results obtained from each stage of our pipeline.

##### Input pages

<table>
	<tr>
		<th>Slime page 4</th>
		<th>Slime page 5</th>
	</tr>
	<tr>
		<td>
			<img src="../images/ocr_results/slime_page_4/input_image.png"></img>
		</td>
		<td>
			<img src="../images/ocr_results/slime_page_5/input_image.png"></img>
		</td>
	</tr>
</table>

##### Initial Tesseract text block extraction (step 1)

Some "good" text blocks from page 4 of slime:

<table>
  <tr>
    <td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_1.png"></img>
  		<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_2.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_5.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_6.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_7.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_8.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_9.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_10.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_11.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_12.png"></img>
  	</td>
  </tr>
	<tr>
    <td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_13.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_15.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_16.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_17.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_18.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_19.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_23.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_24.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_25.png"></img>
  	</td>
  </tr>
	<tr>
    <td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_33.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_35.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_36.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_37.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_38.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_39.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_40.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_41.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_42.png"></img>
  	</td>
  </tr>
</table>
Not all text blocks actually contains texts, here are all bad text blocks from slime page 4:

<table>
  <tr>
    <td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_3.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_14.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_20.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_21.png"></img>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/blocks/block_22.png"></img>
  	</td>
  </tr>
</table>

**Todo: do the same for slime page 5**

##### Learned SIFT descriptor keypoints from text blocks (step 3)

Here are some keypoint descriptors learned from the text blocks that serve as vocabulary (visualized using open cv)

<table>
  <tr>
    <td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/1.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/2.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/5.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/6.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/7.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/8.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/10.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/11.png"></img>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/block_keypoints/12.png"></img>
    </td>
  </tr>
</table>

**Todo: do the same for slime page 5**

##### Find good SIFT matches in input image and cluster using MeanShift (steps 4, 5, 6)

Red rectangles are sift mathces while green rectangles are match cluster centers. Grey rectangles are match centers discarded after thresholding on matched instances in each center.

<table>
	<tr>
		<th>Slime page 4 matches</th>
		<th>Slime page 5 matches</th>
	</tr>
	<tr>
		<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/matches_from_sift.png"></img>
		</td>
		<td>
			<img src="../images/ocr_results/slime_page_5/sift_ocr/matches_from_sift.png"></img>
		</td>
	</tr>
</table>

##### Extracted speech bubble masks (steps 7, 8, 9)

Speech bubble masks extracted from slime page 4

<table>
  <tr><th>Good</th></tr>
	<tr>
		<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/2.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/3.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/5.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/6.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/7.png"></img>
		</td>
</tr>
<tr><th>Bad</th></tr>
	<tr>
		<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/1.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/9.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_4/sift_ocr/masks/10.png"></img>
		</td>
</table>

Speech bubble masks extracted from slime page 5

<table>
  <tr><th>Good</th></tr>
	<tr>
		<td>
			<img src="../images/ocr_results/slime_page_5/sift_ocr/masks/1.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_5/sift_ocr/masks/2.png"></img>
		</td>
<td>
			<img src="../images/ocr_results/slime_page_5/sift_ocr/masks/3.png"></img>
		</td>
</tr>
</table>

##### Mask, crop, then detect text in speech bubbles (steps 10, 11, 12)

Results from slime page 4

<table>
  <tr>
    <td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/masked/2.png"></img>
    </td>
<td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/masked/3.png"></img>
    </td>
<td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/masked/5.png"></img>
    </td>
<td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/masked/6.png"></img>
    </td>
<td>
      <img src="../images/ocr_results/slime_page_4/sift_ocr/masked/7.png"></img>
    </td>
  </tr>
</table>

Results from slime page 5

<table>
  <tr>
    <td>
      <img src="../images/ocr_results/slime_page_5/sift_ocr/masked/1.png"></img>
      <img src="../images/ocr_results/slime_page_5/sift_ocr/masked/2.png"></img>
      <img src="../images/ocr_results/slime_page_5/sift_ocr/masked/3.png"></img>
    </td>
  </tr>
</table>

##### Final Results 

Final results compared with baseline. Baseline is the text blocks detected by directly running Tesseract OCR on the input image.

Slime page 4 side by side comparison:

<table>
  <tr>
    <th>Baseline</th>
    <th>SIFT-OCR (Ours)</th>
  </tr>
  <tr>
    <td>
    	<img src="../images/ocr_results/slime_page_4/ocr_baseline.png"></img>
		</td>
		<td>
    	<img src="../images/ocr_results/slime_page_4/sift_ocr/ocr_result_grouped.png"></img>
		</td>
  </tr>
</table>

Slime page 5 side by side comparison:

<table>
  <tr>
    <th>Baseline</th>
    <th>SIFT-OCR (Ours)</th>
  </tr>
	<tr>
    <td>
    	<img src="../images/ocr_results/slime_page_5/ocr_baseline.png"></img>
		</td>
		<td>
    	<img src="../images/ocr_results/slime_page_5/sift_ocr/ocr_result_grouped.png"></img>
		</td>
  </tr>
</table>



#### Parameters

