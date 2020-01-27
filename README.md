# [Input Similarity from the Neural Network Perspective](https://papers.nips.cc/paper/8775-input-similarity-from-the-neural-network-perspective)

Work in progress.

This is an implementation of the similarity metrics from the above paper.

The basic idea is to create a cosine similarity from an inner product of the gradients of a neural network.

Instead of following the paper, I train autoencoders and use them to measure similarity.

### Basic command usage

```commandline
train.py --config config/mnist_small.yaml --display 10
```
Train an autoencoder on mnist, displaying images every 10 batches

```commandline
similarity.py --config config/mnist_small.yaml --load <path to your trained model> 
```
Uses the trained model to calculate cosine distance between images in the training set

### Configuration

Configuration flags can be specified in argparse parameters, or in yaml files, or in both.

--config parameter is used to specify a yaml file to load parameters from.  The yaml file contents will be added to the 
argparse namespace object.

Precedence is
* Arguments from command line
* Arguments from the config file
* Default value if specified in config.py

Yaml files can contain nested name-value pairs and they will be flattened

```yaml
dataset:
  name: celeba
  train_len: 10000
  test_len: 1000
```

will be flattened to argparse arguments

```
--dataset_name celeba
--dataset_train_len 10000
--dataset_test_len: 1000
```