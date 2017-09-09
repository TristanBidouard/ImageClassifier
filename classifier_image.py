from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import goslate
import cv2
from decimal import *

FLAGS = None

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
gs = goslate.Goslate()

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Image genearated
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  #Create image
  img = cv2.imread(image, cv2.IMREAD_COLOR)
  img = windows(img)

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    i = 0
    for node_id in top_k:
      i = i + 1
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]

      #Terminal display
      #human_string_fr = gs.translate(human_string, 'fr')
      #l = len(human_string)
      #print('[%d] - %s (%.2f%%)' % (i, human_string, score*100))
      print("\n")
      #print('[%d] - %s (%s)' % (i, human_string, human_string_fr))
      print('[%d] - %s' % (i, human_string))
      display(score)
      print('  (%.2f%%)' % (score*100))

      #Image display
      img = label(img, human_string, i, score)

  return img


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def windows(img):

  """

  Create the result windows on the image

  x1,y1 ------
  |          |
  |          |
  |          |
  --------x2,y2
  """

  height, width, _ = img.shape

  #print("IMAGE(%dx%d)" % (width, height))

  windows_height = 120
  windows_width = 350
  x1 = width - (windows_width + 20)
  y1 = height - (windows_height + 20)
  x2 = width - 20
  y2 = height - 20
  fill_color = (255,255,255) 
  contour_color = (0,150, 0)

  #print("WINDOWS")
  #print("X1 : %d  Y1 : %d" % (x1, y1))
  #print("X2 : %d  Y2 : %d" % (x2, y2))

  cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
  cv2.rectangle(img, (x1, y1), (x2, y2), contour_color, 2)

  return img

def label(img, text, i, score):

  """
  Create the label and the rate of the results 
  """
  text = text.split(',')[0]

  font = cv2.FONT_HERSHEY_DUPLEX
  font_scale = 0.4
  text_thickness = 1
  text_width, text_height = cv2.getTextSize(text, font, font_scale, text_thickness)[0]

  height, width, _ = img.shape
  windows_height = 120
  windows_width = 350
  
  space = (windows_height/5) - text_height
  space = int(round(space))
  #print("SPACE =" , space)

  x_label = (width - (windows_width + 20)) + 5

  if i == 1:
    y_label = (height - (windows_height + 20)) + i * (int(round(space/2)) + text_height)
  else:
    y_label = (height - (windows_height + 20)) + (i-1) * (space + text_height) + (int(round(space/2)) + text_height)

  color_true_rect = (0,150,0)
  color_false_rect = (0,0,150)

  x1_true_rect = (width - (windows_width + 20)) + windows_width/2
  x1_true_rect = int(round(x1_true_rect))
  y1_rect = y_label - text_height
  x2_rect = width - 25
  y2_rect = y_label
  x1_false_rect = (width - (windows_width + 20)) + windows_width/2 + score*(x2_rect - x1_true_rect)
  x1_false_rect = int(round(x1_false_rect))

  x_rate = x1_true_rect + 5
  y_rate = y_label
  color_text_rate = (255,255,255)
  rate = str(Decimal(score*100).quantize(Decimal('1.00')))

  #print("TEXT : %s" % text)
  #print("X1 : %d  Y1 : %d" % (x_label, y_label))

  if i == 1:
    text_color = (0,150, 0)
  else:
    text_color = (0, 0, 150)

  text_width, text_height = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
  cv2.putText(img, text, (x_label, y_label), font, font_scale, text_color , text_thickness)

  cv2.rectangle(img, (x1_true_rect, y1_rect), (x2_rect, y2_rect), color_true_rect, -1)
  cv2.rectangle(img, (x1_false_rect, y1_rect), (x2_rect, y2_rect), color_false_rect, -1)

  text_width, text_height = cv2.getTextSize(str(score), font, font_scale, text_thickness)[0]
  cv2.putText(img, rate+"%", (x_rate, y_rate), font, font_scale, color_text_rate, text_thickness)

  return img


def display(num):

  """
  Terminal display
  """
  scale = 150
  num_scale = round(num * scale)
  """i = 0
  while i < l:
    i = i + 1
    print(" ", end='')"""

  i = 0
  print("[", end='')
  while i < num_scale:
    i = i + 1
    print("#", end='')
  j = 0
  while j < scale - i:
    j = j + 1
    print(".", end='')
  print("]", end='')




def main(_):

  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  img = run_inference_on_image(image)
  print("\n\n")
  cv2.imshow('image',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
