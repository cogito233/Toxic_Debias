import logging
import os
import csv
import pandas as pd

from transformers import is_tf_available
from transformers import DataProcessor, InputExample, InputFeatures

class ToxicTransProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def read_csv(self, input_file, quotechar='"'):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            number_line = 0
            li = []
            for i in csv.reader(f, delimiter=",", quotechar=quotechar):
                li.append(i)
                number_line+=1
                if (number_line==20000):
                    return li
            return li

    def read_txt(self, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(f)

    def get_train_examples(self, data_dir):
        # 用来获得变换后的example; 暂时应该没用？
        # TODO
        """See base class."""
        return self._create_train_examples(self.read_csv(os.path.join(data_dir,'train.csv')), "train")
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(self.read_csv(os.path.join(data_dir, "test_public_expanded.csv")), "dev")

    def get_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(self.read_csv(data_dir), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[0:2000]):
            if i == 0:
                continue
            guid = "%s-%s" % (line[0], set_type)
            text_a = line[1]
            label = str(0 if line[12]=="0.0" else 1)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_train_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[0:10000]):
            if i == 0:
                continue
            guid = "%s-%s" % (line[0], set_type)
            text_a = line[2]
            label = str(0 if line[1]=="0.0" else 1)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

if __name__ == "__main__":
    print(ToxicTransProcessor().get_examples("/cluster/project/sachan/zhiheng/datasets/jigsaw-unintended-bias-in-toxicity-classification/test_public_expanded.csv")[5])

