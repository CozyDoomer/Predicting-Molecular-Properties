# Predicting Molecular Properties

Kaggle competition: https://www.kaggle.com/c/champs-scalar-coupling

In this competition, you will develop an algorithm that can predict the magnetic interaction between two atoms in a molecule 
(i.e., the scalar coupling constant).

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| structured data (__graph based__) | log mean average error | regression |

Solved using [lightgbm](https://github.com/microsoft/LightGBM) and a [message passing neural network](https://arxiv.org/pdf/1704.01212.pdf).

## Notes

A lot of additional data that is not usable directly because it's not contained in the test set.

Domain knowledge about atom interaction in molecules was really important (to a certain degree).

Most of the features were calculated using [rdkit](https://www.rdkit.org/docs/GettingStartedInPython.html) and [openbabel](http://openbabel.org/docs/current/UseTheLibrary/Python.html).

## Score 
__top 2%__

| leaderboard   | score | placement |
|----------|:-------------:|---------:|
| public | -2.37190 | __43/2757__ |
| private | -2.36477 | __42/2757__ |
