# Predicting Molecular Properties

[Kaggle competition link](https://www.kaggle.com/c/champs-scalar-coupling)

[Solution writeup](https://www.kaggle.com/c/champs-scalar-coupling/discussion/106263)

In this competition, you will develop an algorithm that can predict the magnetic interaction between two atoms in a molecule 
(i.e., the scalar coupling constant).

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| structured data (__graph based__) | log mean average error | regression |

Started working on this competition using [lightgbm](https://github.com/microsoft/LightGBM) and then used a modified implementation of a [message passing neural network](https://arxiv.org/pdf/1704.01212.pdf).

## Notes

A lot of additional data that is not usable directly because it's not contained in the test set.

Domain knowledge about atom interaction in molecules was really important (to a certain degree).

Most of the features were calculated using [rdkit](https://www.rdkit.org/docs/GettingStartedInPython.html) and [openbabel](http://openbabel.org/docs/current/UseTheLibrary/Python.html).

### local validation for message passing neural network

#### Per coupling type:
- 1JHC: -1.371
- 2JHC: -2.229
- 3JHC: -1.975
- 1JHN: -1.538
- 2JHN: -2.504
- 3JHN: -2.517
- 2JHH: -2.501
- 3JHH: -2.383 

average local log mae: -2.12

## Placement
__top 2%__

| leaderboard   | score | placement |
|----------|:-------------:|---------:|
| public | -2.37190 | __43/2757__ |
| private | -2.36477 | __42/2757__ |
