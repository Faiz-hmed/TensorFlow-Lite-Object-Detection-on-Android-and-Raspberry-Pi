import tensorflow as tf
import numpy as np

def calc_nms(boxes, classes, scores):
    selected_indices = tf.image.non_max_suppression(
      boxes=boxes, scores=scores, max_output_size=70, iou_threshold=0.45 ,score_threshold=0.45)
    boxes = tf.gather(boxes, selected_indices)
    scores = tf.gather(scores, selected_indices)

    return boxes, classes, scores

def trim_boxes(boxes,classes,scores,imH,imW):
  res=[]
  for i in range(len(boxes)):
    box = boxes[i]
    ymin = int(max(1,(box[0] * imH)))
    xmin = int(max(1,(box[1] * imW)))
    ymax = int(min(imH,(box[2] * imH)))
    xmax = int(min(imW,(box[3] * imW)))
      
    for j in range(len(boxes)):

        small_box=boxes[j]
        y1min = int(max(1,(small_box[0] * imH)))
        x1min = int(max(1,(small_box[1] * imW)))
        y1max = int(min(imH,(small_box[2] * imH)))
        x1max = int(min(imW,(small_box[3] * imW)))
        if ymin<y1min and xmin<x1min and ymax>y1max and xmax>x1max and classes[i]==classes[j]:
            if i not in res:
                res.append(i)

    boxes = np.delete(boxes,res,0)
    scores = np.delete(scores,res,0)
    classes = np.delete(classes,res,0)

    return boxes, scores, classes