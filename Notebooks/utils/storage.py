import os

from google.cloud import storage

def get_checkpoint_uri(bucket_name, path, initial_epoch=None):
  if initial_epoch is not None:
    path = os.path.join(path, f'{initial_epoch:03d}')

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blobs = storage_client.list_blobs(bucket,
                                    prefix=path,
                                    fields='items(name),nextPageToken')

  if initial_epoch is not None:
    try:
      next(blobs)
    except:
      return ""
    else:
      return os.path.join("gs://", bucket_name, path)

  initial_epoch = 0
  for blob in blobs:
    name = os.path.basename(blob.name)
    if not name:
      continue
    name,_ = os.path.splitext(name)
    try:
      name = int(name)
    except:
      continue
    if name > initial_epoch:
      initial_epoch = name

  return os.path.join("gs://", bucket_name, path, f'{initial_epoch:03d}') if initial_epoch > 0 else ""