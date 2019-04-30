from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class ProgrammingMethodNamePy(text_problems.Text2TextProblem):
    """Predict method name based on method code in Python."""

    # The locations of the train, dev, and test set.
    GH = "https://raw.githubusercontent.com/EdinburghNLP/code-docstring-corpus/master/V2/repo_split"
    FILENAME = "{}/repo_split.parallel_declbodies.{}"
    DATA_URLS = {
        problem.DatasetSplit.TRAIN: FILENAME.format(GH, "train"),
        problem.DatasetSplit.EVAL: FILENAME.format(GH, "valid"),
        problem.DatasetSplit.TEST: FILENAME.format(GH, "test"),
    }

    @staticmethod
    def _extract_filename_from_url(url):
        # Ex: TRAIN_URL --> repo_split.parallel_declbodies.train

        # Get everything from the last / onwards.
        return os.path.basename(url)

    @property
    def approx_vocab_size(self):
        return 2**13  # ~8k

    @property
    def is_generate_per_split(self):
        # Return True since we already have the train and the valid set separated out.
        return True

    def maybe_download_dataset(self, tmp_dir, dataset_split):
        """Downloads the appropriate dataset file and returns its path."""
        # Get the dataset url for the split requested.
        url = self.DATA_URLS.get(dataset_split, None)

        # Sanity check.
        if url is None:
            tf.logging.fatal(
                "Unknown dataset_split passed: {}".format(dataset_split))
            return

        # Download the data, if it doesn't already exist.
        return generator_utils.maybe_download(tmp_dir,
                                              self._extract_filename_from_url(
                                                  url),
                                              url)

    def preprocess_input(self, input):
        """Apply some preprocessing to the input.

        For instance, remove space/tabs.

        Args:
          input (str): code source content

        Returns:
          the pre-processed string content
        """
        return input.replace(" DCNL", "\n").replace(" DCSP ", " ")

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir

        # Download the data, if it doesn't already exist.
        downloaded_filepath = self.maybe_download_dataset(
            tmp_dir, dataset_split)

        # Decompress the file and iterate through it.
        tf.logging.debug("Reading {}".format(downloaded_filepath))
        with open(downloaded_filepath, "r") as data_fp:
            for line in data_fp:
                method_name_body = line.split(":", 1)

                method_name = method_name_body[0]
                if method_name.startswith("def "):
                    method_name = method_name[4:]
                if "(" in method_name:
                    method_name = method_name[:method_name.index("(")]

                method_body = method_name_body[1]
                method_body = self.preprocess_input(method_body).strip()
                yield {
                    "inputs": method_body,
                    "targets": method_name,
                }
