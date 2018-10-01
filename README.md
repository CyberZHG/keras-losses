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
