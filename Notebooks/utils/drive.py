import io
import os

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from importlib.machinery import SourceFileLoader


class MyDrive():
  def __init__(self, service, root=""):
    self.service = service
    self.root = root
    self.cache = {"":"root"}


  def get_id(self, file_path):
    file_path = os.path.normpath(file_path)
    file_path = os.path.join(self.root, file_path)
    stack = []
    while True:
      if file_path in self.cache:
        if not stack:
          return self.cache[file_path]
      else:
        file_path, tail = os.path.split(file_path)
        stack.append(tail)
        continue
      
      name = stack.pop()
      response = self.service.files().list(q="name='{}' and "
                                             "'{}' in parents and "
                                             "trashed = false".format(name,
                                                                      self.cache[file_path]),
                                           spaces='drive',
                                           fields='files(id)').execute()
      files = response.get('files', [])
      if not files:
        return ""

      file_path = os.path.join(file_path, name)
      self.cache[file_path] = files[0].get('id')


  def download(self, file_path, force = True):
    if not force and os.path.isfile(file_path):
      return
    file_id = self.get_id(file_path)
    request = self.service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    while True:
      _, done = downloader.next_chunk()
      if done is True:
        break
    path = os.path.dirname(file_path)
    os.makedirs(path, exist_ok = True)
    with open(file_path, "wb") as f:
      f.write(file.getbuffer())


  def load_module(self, file_path):
    self.download(file_path)
    name = os.path.basename(file_path)
    return SourceFileLoader(name, file_path).load_module()


  def create_folder(self, folder_path, create_path=False):
    folder_path = os.path.normpath(folder_path)
    folder_path = os.path.join(self.root, folder_path)
    stack = []
    while True:
      if folder_path in self.cache:
        if not stack:
          return self.cache[folder_path]
      else:
        folder_path, tail = os.path.split(folder_path)
        stack.append(tail)
        continue
      
      name = stack.pop()
      response = self.service.files().list(q="mimeType='{}' and"
                                             "name='{}' and "
                                             "'{}' in parents and "
                                             "trashed = false".format('application/vnd.google-apps.folder',
                                                                      name,
                                                                      self.cache[folder_path]),
                                           spaces='drive',
                                           fields='files(id)').execute()
      files = response.get('files', [])
      if not files:
        if create_path:
          folder = self.service.files().create(
            body={
              'name': name,
              'parents': [self.cache[folder_path]],
              'mimeType': 'application/vnd.google-apps.folder'
            }, fields='id').execute()
          files.append(folder)
        else:
          return ""

      folder_path = os.path.join(folder_path, name)
      self.cache[folder_path] = files[0].get('id')


  def copy_file(self, from_file, to_file, create_path=False, force=False):
    from_file_id = self.get_id(from_file)
    to_file_folder, to_file_name = os.path.split(to_file)
    parent = self.get_id(to_file_folder)
    if not parent:
      if create_path:
        parent = self.create_folder(to_file_folder, True)
      else:
        return ""
    if self.get_id(to_file):
      if force:
        self.delete(to_file)
      else:
        return self.get_id(to_file)

    file = self.service.files().copy(fileId=from_file_id,
      body={
        'name': to_file_name,
        'parents': [parent]
      }, fields='id').execute()

    return self.get_id(to_file)


  def upload(self, from_file, to_file, create_path=False, force=False):
    to_file_folder, to_file_name = os.path.split(to_file)
    parent = self.get_id(to_file_folder)
    if not parent:
      if create_path:
        parent = self.create_folder(to_file_folder, True)
      else:
        return ""
    if self.get_id(to_file):
      if force:
        self.delete(to_file)
      else:
        return ""

    media = MediaFileUpload(from_file,
                            mimetype='*/*',
                            resumable=False)
    file = self.service.files().create(
      body={
        'name': to_file_name,
        'parents': [parent],
        'mimeType': '*/*'
    }, media_body=media, fields='id').execute()

    return self.get_id(to_file)

  
  def delete(self, file):
    file_id = self.get_id(file)
    self.service.files().delete(fileId=file_id).execute()
    self.cache = {"":"root"}