So far, we have demonstrated the outstanding performance of SIFT-OCR compared to the baseline method in the task of extracting
text from manga speech bubbles. This is the most difficult hurdle to get over in our pipeline. It required huge amount of innovation
and effort from our team. That said, the algorithm is not without its defects - it has many tunable parameters, the effects of some
are not fully understood. In addition, there are many other tasks down the pipeline, including stitching sentences together from
detected text blocks, machine translation, and hypothesizing new bounding boxes for embedding the translated text. No matter how well
an automated system performs, mistakes are inevitable - to make our system production ready, we'll also need to add UI
for humans to optionally step in at every stage of the pipeline. We'll choose some of the tasks listed above to complete in the future.