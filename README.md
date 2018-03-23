# BinaryNet-on-tensorflow
binary weight neural network implementation on tensorflow

This is an implementation code for reproducing BNN

## How to run
```
python mnist.py
python cifar10.py
```

## Accuracy
| DataSet | accuracy |
|---------|----------|
| MNIST   |  99.04%  |
| Cifar10 |  86.18%  |

## Different between paper
layer-wise learning rate, paper is layer_lr = 1./sqrt(1.5 / (num_inputs + num_units)), my implement is layer_lr / 4

## Ref
```
[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1"](http://arxiv.org/abs/1602.02830),
Matthieu Courbariaux, Yoshua Bengio
```
and
[An implemtation of binaryNet for Keras](https://github.com/DingKe/nn_playground/tree/master/binarynet)