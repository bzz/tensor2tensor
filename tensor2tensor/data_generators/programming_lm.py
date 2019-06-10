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
_HOST = "https://storage.googleapis.com/dataset.srcd.host/"


def _maybe_download_corpus(tmp_dir):
  """Download and unpack the corpus.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    The list of names of files.
  """
  preprocessed_url = _HOST + "java_projects_processed.txt.xz"
  preprocessed_parts_url = _HOST + "java_projects_processed_parts.tar.xz"

  compressed_parts_filename = os.path.basename(preprocessed_parts_url)
  compressed_parts_filepath = os.path.join(tmp_dir, compressed_parts_filename)

  decompressed_filepath = os.path.join(tmp_dir,
                                       os.path.basename(preprocessed_url))[:-3]
  parts_file_prefix = decompressed_filepath + "-part-"
  parts_filepattern = parts_file_prefix + "?????"
  parts_files = sorted(tf.gfile.Glob(parts_filepattern))
  tf.logging.info("Listing %s, %d found", parts_filepattern, len(parts_files))
  if parts_files:  # uncompressed parts exist
    return parts_files

  ## try already preprocessed & part(itioned) dataset
  if not tf.gfile.Exists(compressed_parts_filepath):
    tf.logging.info(
        "Archive {} not found, downloading".format(compressed_parts_filepath))
    compressed_parts_filepath = generator_utils.maybe_download(
        tmp_dir, compressed_parts_filename, preprocessed_parts_url)

    tf.logging.info("Decompressing {}".format(compressed_parts_filepath))
    assert not subprocess.call(
        ["tar", "Jxf", compressed_parts_filepath, "-C", tmp_dir],
        env=dict(os.environ, XZ_OPT="-T0"))
    parts_files = sorted(tf.gfile.Glob(parts_filepattern))
    tf.logging.info("Listing %s, %d found", parts_filepattern, len(parts_files))

  if not parts_files:
    tf.logging.info("Cann't use partitioned dataset, tring to partition")
    return _maybe_download_uncompress_partition(preprocessed_url, tmp_dir)
  return parts_files


def _maybe_download_uncompress_partition(preprocessed_url, tmp_dir):
  """Downloads, unpacks, partitions the single file of a pre-processed corpus.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    The list of names of files.
  """
  compressed_filename = os.path.basename(preprocessed_url)
  compressed_filepath = os.path.join(tmp_dir, compressed_filename)
  ## download
  if not tf.gfile.Exists(compressed_filepath):
    tf.logging.info(
        "Archive {} not found, downloading".format(compressed_filepath))
    compressed_filepath = generator_utils.maybe_download(
        tmp_dir, compressed_filename, preprocessed_url)

  decompressed_filepath = compressed_filepath[:-3]
  parts_file_prefix = decompressed_filepath + "-part-"
  parts_filepattern = parts_file_prefix + "?????"
  parts_files = sorted(tf.gfile.Glob(parts_filepattern))
  tf.logging.info("Listing %s, %d found", parts_filepattern, len(parts_files))

  if not parts_files:
    ## un-compress downloaded .xz
    if not tf.gfile.Exists(decompressed_filepath):
      tf.logging.info("Decompressing {}".format(compressed_filepath))
      assert not subprocess.call(["xz", "-dk", "-T0", compressed_filepath])
    assert tf.gfile.Exists(decompressed_filepath)

    ## partition single big file (12Gb) into multiple small text files
    tf.logging.info("Partition {} to 4Mb files".format(decompressed_filepath))
    assert not subprocess.call([
        "split", "--line-bytes=4M", "--suffix-length=5", "--numeric-suffixes",
        decompressed_filepath, parts_file_prefix
    ])
    tf.gfile.Remove(decompressed_filepath)
    parts_files = sorted(tf.gfile.Glob(parts_filepattern))

  return parts_files


@registry.register_problem
class ProgrammingLmJava32k(text_problems.Text2SelfProblem):
  """
  Data for Javafrom 2013.msrconf.org/slides/msr13-allamanis2.pdf
  Dataset http://groups.inf.ed.ac.uk/cup/javaGithub/
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
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
  without regard to file boundaries.
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


def _maybe_download_splitted_corpus(tmp_dir):
  """Downloads and unpacks files for
      - pre-processed (2M files -> 1 file of 12Gb)
      - pre-splitted (train/validation/test without duplicates)
      - pre-partitioned (each split in multiple 4Mb -part-????)
     corpus.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    The list of names of files.
  """
  basename = "java_projects_processed_"
  url = _HOST + basename + "split_parts.tar.xz"

  compressed_parts_filename = os.path.basename(url)
  compressed_parts_filepath = os.path.join(tmp_dir, compressed_parts_filename)

  parts_file_prefix = os.path.join(tmp_dir, basename + "*.txt" + "-part-")
  parts_filepattern = parts_file_prefix + "?????"
  parts_files = sorted(tf.gfile.Glob(parts_filepattern))
  tf.logging.info("Listing %s, %d found", parts_filepattern, len(parts_files))
  if parts_files:  # uncompressed parts exist
    return parts_files

  ## try already preprocessed & part(itioned) dataset
  if not tf.gfile.Exists(compressed_parts_filepath):
    tf.logging.info(
        "Archive {} not found, downloading".format(compressed_parts_filepath))
    compressed_parts_filepath = generator_utils.maybe_download(
        tmp_dir, compressed_parts_filename, url)

  tf.logging.info("Decompressing {}".format(compressed_parts_filepath))
  assert not subprocess.call(
      ["tar", "Jxf", compressed_parts_filepath, "-C", tmp_dir],
      env=dict(os.environ, XZ_OPT="-T0"))

  parts_files = sorted(tf.gfile.Glob(parts_filepattern))
  tf.logging.info("Listing %s, %d found", parts_filepattern, len(parts_files))
  return parts_files


@registry.register_problem
class ProgrammingLmJava32kSplitChopped(text_problems.ChoppedTextProblem):
  """
  Pre-splitted version that relies on accurate splits duing pre-processing
  (see preprocess-java.go)

  All files are chopped arbitrarily into sequences of length 256 tokens,
  without regard to file boundaries.
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def is_generate_per_split(self):
    return True

  @property
  def max_dev_chars(self):
    return 10**8  # 100 mb

  @property
  def max_chars_for_vocab(self):
    """Number of characters of training data to use for generating vocab."""
    return 10**8 # 100 Mb

  @property
  def dataset_splits(self):
    """Number of output shards for each split.

    Returns:
      A dict containing splits information.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 20,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def train_text_filepaths(self, tmp_dir):
    all_files = _maybe_download_splitted_corpus(tmp_dir)
    return [f for f in all_files if "_train.txt-part-" in f]

  def dev_text_filepaths(self, tmp_dir):
    all_files = _maybe_download_splitted_corpus(tmp_dir)
    return [f for f in all_files if "_val.txt-part-" in f]

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 256

@registry.register_problem
class ProgrammingLmJava32kSplitChoppedCharacter(ProgrammingLmJava32kSplitChopped):
  """Programming LM, character-level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def filepath_to_unicode_strings(self, filepath):
    """Read text out of an input file.
    Preprocess, to restor newlines

    Args:
      filepath: a string
    Yields:
      unicode strings.
    """
    f = tf.gfile.Open(filepath)
    b = f.read()
    c = b.replace('\\n', '\n')
    del b
    yield text_encoder.to_unicode_ignore_errors(c)

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

# - [x] char-level model + keep '\n' (custom .filepath_to_unicode_strings())

# - [ ] a custom tokenizer for code that preserves whitespaces

# - discard subtokens by freq <= 3
#   `SubwordTextEncoder.build_to_target_size(.., min_val=3, ...)`
#
# - hyperparams: vocab size 25k, seq length 200, vocab learning effect
#   To limit the number of samples the vocab generation
#   looks at, override `self.max_samples_for_vocab

# - models: Transformer-XL (recurrance, memory), Universla Transformer
