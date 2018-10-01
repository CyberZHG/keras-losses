# Keras Losses

[![Travis](https://travis-ci.org/CyberZHG/keras-losses.svg)](https://travis-ci.org/CyberZHG/keras-losses)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-losses/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-losses)

Some loss functions in Keras.

## Install

```bash
python setup.py install
```

## Usage

### Ranking Loss

![](https://user-images.githubusercontent.com/853842/46719694-29ed0400-cca1-11e8-9be5-9bf298952d90.png)

```python
from keras_losses import get_ranking_loss

ranking_loss = get_ranking_loss(gamma=2.0, mp=2.5, mn=0.5)
```

### Weighted Categorical Cross-Entropy Loss

![](https://user-images.githubusercontent.com/853842/46720488-3eca9700-cca3-11e8-845d-b44adc31df74.png)

```python
from keras_losses import get_weighted_categorical_crossentropy, get_weighted_sparse_categorical_crossentropy

weighted_loss = get_weighted_categorical_crossentropy(weights=[0.8, 0.1, 0.2, 0.3, 0.4])
weighted_loss = get_weighted_sparse_categorical_crossentropy(weights=[0.8, 0.1, 0.2, 0.3, 0.4])
```
