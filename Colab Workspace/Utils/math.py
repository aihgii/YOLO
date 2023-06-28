import tensorflow as tf


def intersection_over_union(box1, box2):

  box1_x1 = box1[..., 0:1] - (box1[..., 2:3] / 2)
  box1_y1 = box1[..., 1:2] - (box1[..., 3:4] / 2)
  box1_x2 = box1[..., 0:1] + (box1[..., 2:3] / 2)
  box1_y2 = box1[..., 1:2] + (box1[..., 3:4] / 2)
  box2_x1 = box2[..., 0:1] - (box2[..., 2:3] / 2)
  box2_y1 = box2[..., 1:2] - (box2[..., 3:4] / 2)
  box2_x2 = box2[..., 0:1] + (box2[..., 2:3] / 2)
  box2_y2 = box2[..., 1:2] + (box2[..., 3:4] / 2)

  x1 = tf.math.maximum(box1_x1, box2_x1)
  y1 = tf.math.maximum(box1_y1, box2_y1)
  x2 = tf.math.minimum(box1_x2, box2_x2)
  y2 = tf.math.minimum(box1_y2, box2_y2)

  dx = tf.math.maximum(x2 - x1, 0)
  dy = tf.math.maximum(y2 - y1, 0)
  intersection = dx * dy
  
  box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
  box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
  union = box1_area + box2_area - intersection
  
  iou = tf.where(tf.math.greater(union, 0.0), intersection / union, 0.0)

  return iou
