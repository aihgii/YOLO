import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.__internal__ import layers

def h_flip(image, labels):
  flipped_image = tf.image.flip_left_right(image)
  flipped_labels = []
  for c, x, y, w, h in labels:
    flipped_labels.append((c, 1-x, y, w, h))

  return flipped_image, flipped_labels

def crop_on_boxes(image, labels, min_ratio, padding):
  image_shape = tf.shape(image)
  width = int(image_shape[0])
  height = int(image_shape[1])

  crop_left_x = width
  crop_top_y = height
  crop_right_x = 0
  crop_bottom_y = 0
  cropped_labels = []
  for c, x, y, w, h in labels:
    # transforming the box coordinates to absolute values with respect to the top-left corner
    box_left_x = tf.math.round((x - (w / 2)) * width)
    box_top_y = tf.math.round((y - (h / 2)) * height)
    box_width = tf.math.round(w * width)
    box_height = tf.math.round(h * height)
    box_right_x = box_left_x + box_width
    box_bottom_y = box_top_y + box_height

    # getting outer bounding box for all the boxes + padding
    crop_left_x = tf.minimum(crop_left_x, box_left_x - padding)
    crop_top_y = tf.minimum(crop_top_y, box_top_y - padding)
    crop_right_x = tf.maximum(crop_right_x, box_right_x + padding)
    crop_bottom_y = tf.maximum(crop_bottom_y, box_bottom_y + padding)

  # don't allow to exceed the minimum ratio
  crop_width = crop_right_x - crop_left_x
  h_ratio = crop_width / width
  if h_ratio < min_ratio:
    min_width = tf.math.round(width * min_ratio)
    extra_padding = min_width - crop_width
    crop_width = min_width
    crop_left_x -= extra_padding // 2
    crop_right_x += extra_padding // 2
  crop_height = crop_bottom_y - crop_top_y
  v_ratio = crop_height / height
  if v_ratio < min_ratio:
    min_height = tf.math.round(height * min_ratio)
    extra_padding = min_height - crop_height
    target_height = min_height
    crop_top_y -= extra_padding // 2
    crop_bottom_y += extra_padding // 2

  # clipping the outer box to the image edges if we got outside
  crop_left_x = tf.maximum(crop_left_x, 0)
  crop_top_y = tf.maximum(crop_top_y, 0)
  crop_right_x = tf.minimum(crop_right_x, width)
  crop_bottom_y = tf.minimum(crop_bottom_y, height)
  crop_width = crop_right_x - crop_left_x
  crop_height = crop_bottom_y - crop_top_y

  for c, x, y, w, h in labels:
    x = (tf.math.round(x * width) - crop_left_x) / crop_width
    y = (tf.math.round(y * height) - crop_top_y) / crop_height
    w *= width / crop_width
    h *= height / crop_height
    cropped_labels.append((c,x,y,w,h))

  cropped_image = tf.image.crop_to_bounding_box(image,
                                                int(crop_top_y), int(crop_left_x),
                                                int(crop_height), int(crop_width))

  cropped_image = tf.image.resize(cropped_image, [width, height])

  return cropped_image, cropped_labels


class RandomSaturation(layers.BaseRandomLayer):

  def __init__(self, lower, upper, seed=None, **kwargs):
    super().__init__(seed=seed, force_generator=True, **kwargs)

    if lower < 0:
      raise ValueError(
        "argument `lower` cannot have a negative value"
        f"Received: lower={lower}"
      )
    if upper < lower:
      raise ValueError(
        "argument `upper` must be greater or equal to argument `lower`"
        f"Received: lower={lower}, upper={upper}"
      )

    self._lower = lower
    self._upper = upper

  def call(self, inputs, training=True):
    if training:
      return self._saturation_adjust(inputs)
    else:
      return inputs

  def _saturation_adjust(self, inputs):
    seed = self._random_generator.make_seed_for_stateless_op()
    if seed is not None:
      output = tf.image.stateless_random_saturation(
        inputs, self._lower, self._upper, seed=seed
      )
    else:
      output = tf.image.random_saturation(
        inputs, self._lower, self._upper,
        seed=self._random_generator.make_legacy_seed()
      )
    output.set_shape(inputs.shape)
    return output