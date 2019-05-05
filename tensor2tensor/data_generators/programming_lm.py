from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

_URL = "http://groups.inf.ed.ac.uk/cup/javaGithub/java_projects.tar.gz"


def _maybe_download_corpus(tmp_dir):
  """Download and unpack the corpus.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    The list of names of files.
  """
  host = "https://storage.googleapis.com/dataset.srcd.host/"
  preprocessed_url = host + "java_projects_processed.txt.xz"
  preprocessed_split_url = host + "java_projects_processed_parts.tar.xz"

  compressed_split_filename = os.path.basename(preprocessed_split_url)
  compressed_split_filepath = os.path.join(tmp_dir, compressed_split_filename)

  decompressed_filepath = os.path.join(tmp_dir,
                                       os.path.basename(preprocessed_url))[:-3]
  split_file_prefix = decompressed_filepath + "-part-"
  split_filepattern = split_file_prefix + "?????"
  split_files = sorted(tf.gfile.Glob(split_filepattern))
  tf.logging.info("Listing %s, %d found", split_filepattern, len(split_files))
  if split_files:
    return split_files

  ## try already preprocessed & splited dataset first
  if not tf.gfile.Exists(compressed_split_filepath):
    tf.logging.info(
        "Archive {} not found, downloading".format(compressed_split_filepath))
    compressed_split_filepath = generator_utils.maybe_download(
        tmp_dir, compressed_split_filename, preprocessed_split_url)

    tf.logging.info("Decompressing {}".format(compressed_split_filepath))
    assert not subprocess.call(
        ["tar", "Jxf", compressed_split_filepath, "-C", tmp_dir],
        env=dict(os.environ, XZ_OPT="-T0"))
    split_files = sorted(tf.gfile.Glob(split_filepattern))
    tf.logging.info("Listing %s, %d found", split_filepattern, len(split_files))

  if not split_files:
    tf.logging.info("Cann't use pre-split dataset, fall back to splitting")
    return _maybe_download_uncompress_split(preprocessed_url, tmp_dir)
  return split_files


def _maybe_download_uncompress_split(preprocessed_url, tmp_dir):
  compressed_filename = os.path.basename(preprocessed_url)
  compressed_filepath = os.path.join(tmp_dir, compressed_filename)
  ## download
  if not tf.gfile.Exists(compressed_filepath):
    tf.logging.info(
        "Archive {} not found, downloading".format(compressed_filepath))
    compressed_filepath = generator_utils.maybe_download(
        tmp_dir, compressed_filename, preprocessed_url)

  decompressed_filepath = compressed_filepath[:-3]
  split_file_prefix = decompressed_filepath + "-part-"
  split_filepattern = split_file_prefix + "?????"
  split_files = sorted(tf.gfile.Glob(split_filepattern))
  tf.logging.info("Listing %s, %d found", split_filepattern, len(split_files))

  if not split_files:
    ## un-compress downloaded .xz
    if not tf.gfile.Exists(decompressed_filepath):
      tf.logging.info("Decompressing {}".format(compressed_filepath))
      assert not subprocess.call(["xz", "-dk", "-T0", compressed_filepath])
    assert tf.gfile.Exists(decompressed_filepath)

    ## split into multiple text files
    tf.logging.info("Splitting {} to 4Mb files".format(decompressed_filepath))
    assert not subprocess.call([
        "split", "--line-bytes=4M", "--suffix-length=5", "--numeric-suffixes",
        decompressed_filepath, split_file_prefix
    ])
    tf.gfile.Remove(decompressed_filepath)
    split_files = sorted(tf.gfile.Glob(split_filepattern))

  return split_files


@registry.register_problem
class ProgrammingLmJava32k(text_problems.Text2SelfProblem):
  """
  Data for Javafrom 2013.msrconf.org/slides/msr13-allamanis2.pdf
  Dataset http://groups.inf.ed.ac.uk/cup/javaGithub/
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def is_generate_per_split(self):
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.

    Returns:
      A dict containing splits information.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def dev_fraction(self):
    return 100  # 12GB/4MB = 3000, take 1%

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    all_files = _maybe_download_corpus(tmp_dir)

    split_files = {
        problem.DatasetSplit.TRAIN: [
            f for i, f in enumerate(all_files) if i % self.dev_fraction != 0
        ],
        problem.DatasetSplit.EVAL: [
            f for i, f in enumerate(all_files) if i % self.dev_fraction == 0
        ],
    }

    files = split_files[dataset_split]
    for filepath in files:
      tf.logging.debug("Reading %s", filepath)
      for line in tf.gfile.Open(filepath):
        txt = text_encoder.native_to_unicode(line)
        yield {"targets": txt}


@registry.register_problem
class ProgrammingLmJava32kPacked(ProgrammingLmJava32k):
  """Packed version for TPU training."""

  @property
  def packed_length(self):
    return 256

  @property
  def vocab_filename(self):
    return ProgrammingLmJava32k().vocab_filename


@registry.register_problem
class ProgrammingLmJava32kChopped(text_problems.ChoppedTextProblem):
  """
  All files are chopped arbitrarily into sequences of length 256 tokens,
  without regard to article boundaries.
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def is_generate_per_split(self):
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.

    Returns:
      A dict containing splits information.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def dev_fraction(self):
    return 100  # 12GB/4MB = 3000, take 1%

  def train_text_filepaths(self, tmp_dir):
    all_files = _maybe_download_corpus(tmp_dir)
    return [f for i, f in enumerate(all_files) if i % self.dev_fraction != 0]

  def dev_text_filepaths(self, tmp_dir):
    all_files = _maybe_download_corpus(tmp_dir)
    return [f for i, f in enumerate(all_files) if i % self.dev_fraction == 0]

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 256

  @property
  def max_chars_for_vocab(self):
    """Number of characters of training data to use for generating vocab."""
    return 1000 * 10**6  # 1 Gb

  @property
  def max_dev_chars(self):
    """Limit dev set to at most this many characters"""
    return 100 * 10**6  # 100 Mb


#TODO:
# - context/sequence lenght? TPU only supports fixed length examples seq!
#   default Transformer TPU: hparams.max_length = 64
# Adaptive batch sizes and sequence lengths are not supported on TPU.
# Instead, every batch has the same sequence length and the same batch size.
# Longer sequences are dropped and shorter ones are padded.
#
# - [x] A _packed option, seq length 256
#   like LanguagemodelDeEnFrRoWiki64kFitbPacked1k or LanguagemodelLm1b32kPacked
#   https://github.com/tensorflow/tensor2tensor/blob/f679aba4a254cb7f5c6cea11f3e431226a269957/tensor2tensor/models/transformer.py#L2342
#
#  Text2TextProblem.packed_length() overrides max_length,
#    Pack multiple examples into a single example of constant length.
#    This is useful for TPU training to reduce the fraction of padding tokens.

# - [x] chopped mode option
#  ChoppedTextProblem is usually used for that and it
#    is alos only one, that is affected by --num_concurrent_processes=N

# - discard subtokens by freq <= 3
#   `SubwordTextEncoder.build_to_target_size(.., min_val=3, ...)`
#
# - hyperparams: vocab size 25k, seq length 200, vocab learning effort
#   To limit the number of samples the vocab generation
#   looks at, override `self.max_samples_for_vocab

# - models: Transformer-XL (recurrance, memory), Universla Transformer
