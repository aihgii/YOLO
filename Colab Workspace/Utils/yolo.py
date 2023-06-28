import tensorflow as tf


def get_boxes(prediction, confidence_threshold, S, B, C):
  """
  Retrieves a list of boxes from YOLO model prediction as a 2D tensor.
  Values per item:
   - idx (sample id)
   - class
   - confidence
   - x coordinate of the bbox
   - y coordinate of the bbox
   - width of the bbox
   - height of the bbox

  Args:
    prediction (tensor): model prediction of shape [batch_size, S, S, C + B * 5]
    confidence_threshold (float): threshold to ignore boxes with lower prediction confidence
    S (integer): grid size (parameter of the model)
    B (integer): predictions per grid cell (parameter of the model)
    C (integer): number of classes (parameter of the model)

  Returns:
    prediction_boxes - tensor of shape [None, 7]
  """

  batch_size = tf.shape(prediction)[0]
  
  classes = tf.slice(prediction, [0,0,0,0], [batch_size,S,S,C])
  classes = tf.math.argmax(classes, axis=-1, output_type=tf.dtypes.int32)
  classes = tf.cast(classes, tf.dtypes.float32)
  classes = tf.reshape(classes, [batch_size, S, S, 1])
  
  boxes = tf.slice(prediction, [0,0,0,C], [batch_size,S,S,B*5])
  boxes = tf.concat([classes, boxes], 3)
  
  count = 0
  prediction_boxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  for item in tf.range(batch_size):
    for i in tf.range(S):
      for j in tf.range(S):
        for b in tf.range(B):
          if (boxes[item][i][j][(1+b*5)]) < confidence_threshold:
            continue
          c = boxes[item][i][j][0]
          conf = boxes[item][i][j][(1+b*5)]
          x = (boxes[item][i][j][(1+b*5)+1] + float(j)) / S
          y = (boxes[item][i][j][(1+b*5)+2] + float(i)) / S
          w = boxes[item][i][j][(1+b*5)+3]
          h = boxes[item][i][j][(1+b*5)+4]
          box = tf.stack([float(item),c,conf,x,y,w,h])
          prediction_boxes = prediction_boxes.write(count, box)
          count += 1

  return prediction_boxes.stack() # idx, class, confidence, x, y, w, h


def get_example(image, labels, S, B, C):
  """
  Creates a tf.train.Example instance from sample image and labels in the format proper for YOLO model training

  Args:
    image (tensor): sample image
    labels (list of tuples): list of bboxes in format (class, x, y, width, height)
    S (integer): grid size (parameter of the model)
    B (integer): predictions per grid cell (parameter of the model)
    C (integer): number of classes (parameter of the model)

  Returns:
    example - tf.train.Example instance
  """
  image = tf.io.serialize_tensor(image).numpy()

  used_cells = set()
  label = [[[0.0 for k in range(C + 5)] for j in range(S)] for i in range(S)]
  for c, x, y, w, h in labels:
    i = int(S * y)
    j = int(S * x)
    x_cell = S * x - j
    y_cell = S * y - i

    if (i,j) in used_cells:
      continue
    used_cells.add((i,j))

    label[i][j][int(c)] = 1.0
    label[i][j][C]      = 1.0
    label[i][j][C + 1]  = x_cell
    label[i][j][C + 2]  = y_cell
    label[i][j][C + 3]  = w
    label[i][j][C + 4]  = h
  label = tf.cast(label, tf.float32)
  label = tf.io.serialize_tensor(label).numpy()

  return tf.train.Example(
    features=tf.train.Features(feature={
      "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
      "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
  }))