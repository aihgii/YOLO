import os
import tensorflow as tf


class Logger(tf.keras.callbacks.Callback):
  def __init__(self, log_dir, experiment_name=None, device='/job:localhost'):
    self.device = device
    self.log_dir = log_dir if experiment_name is None else os.path.join(log_dir, experiment_name)
    self.log_dir_training = os.path.join(self.log_dir, 'training')
    self.log_dir_validation = os.path.join(self.log_dir, 'validation')

    os.makedirs(self.log_dir_training, exist_ok = True)
    os.makedirs(self.log_dir_validation, exist_ok = True)


  def on_epoch_end(self, epoch, logs=None):
    with tf.device(self.device):
      with tf.summary.create_file_writer(self.log_dir_training).as_default():
        #tf.summary.scalar('lr', data=self.model.optimizer.lr, step=epoch)
        for metric in logs.keys():
          if metric.startswith('val_'):
            continue
          tf.summary.scalar(metric, logs[metric], step=epoch)
      with tf.summary.create_file_writer(self.log_dir_validation).as_default():
        for metric in logs.keys():
          if metric.startswith('val_'):
            tf.summary.scalar(metric[4:], logs[metric], step=epoch)
