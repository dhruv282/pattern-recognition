# Neural Network Pattern Detection

The program [detectPattern.py](detectPattern.py) is an implementation of a neural network model that is supposed to recognize pictures of handwritten numbers from [this](https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits-orig.windep.Z) dataset. The `encodeFiles()` function in the program splits creates [input.txt](input.txt) and [target.txt](target.txt) files from [optdigits-orig.windep](optdigits-orig.windep). The file consists of 1797 instances from 43 writers and the training, validation, and testing datasets are split as follows:

* Training dataset: 1300
* Validation dataset: 200
* Testing dataset: 297


## Running the program

[Python 3](https://www.python.org/downloads/) along with the [numpy](https://numpy.org/) library is required to run this program.

```shell
$ python3 detectPattern.py
```