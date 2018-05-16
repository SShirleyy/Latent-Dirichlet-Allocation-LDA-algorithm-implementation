import os, pickle, tarfile, codecs
import numpy as np
from os.path import dirname, exists, expanduser, isdir, join, splitext
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.datasets.base import load_files


archive_path = '20news-bydate.tar.gz'
train_path = '20news-bydate-train'
test_path = '20news-bydate-test'

# tarfile.open(archive_path, "r:gz").extractall()
cache_path = '20news-bydate.pkz'

cache = dict(train=load_files(train_path, encoding='latin1'),
  test=load_files(test_path, encoding='latin1'))
compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
with open(cache_path, 'wb') as f:
  f.write(compressed_content)