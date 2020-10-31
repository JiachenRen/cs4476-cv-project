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

##### Data

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

