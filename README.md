# Predicting-Molecular-Properties

- https://www.kaggle.com/c/champs-scalar-coupling

## Results so far 

Lightgbm, meta-feature kaggle notebook as starting point

### scores for l2 regression

with lambda_l1 = 0.1 and lambda_l2 = 0.3

- CV mean score: -0.4101, std: 0.0009.
- CV mean score: -1.7386, std: 0.0032.
- CV mean score: -0.9993, std: 0.0088.
- CV mean score: -1.6426, std: 0.0079.
- CV mean score: -1.2682, std: 0.0029.
- CV mean score: -1.5983, std: 0.0076.
- CV mean score: -1.0264, std: 0.0039.
- CV mean score: -1.8335, std: 0.0189.

LB score: -1.584

### best mae/l1 regression score

lambda_l1 = 0.1, lambda_l2 = 0.3, factor = np.tanh(variance/20)+1

- CV mean score: -0.3981, std: 0.0016.
- CV mean score: -1.7055, std: 0.0036.
- CV mean score: -0.9291, std: 0.0092.
- CV mean score: -1.6048, std: 0.0103.
- CV mean score: -1.2033, std: 0.0021.
- CV mean score: -1.5529, std: 0.0051.
- CV mean score: -0.9857, std: 0.0026.
- CV mean score: -1.8029, std: 0.0085.

LB score: -1.611

### l1 regression with different params

lambda_l1 = 0.4, lambda_l2 = 0.2, factor = np.tanh(variance/30)+1

- CV mean score: -0.3631, std: 0.0034.
- CV mean score: -1.4712, std: 0.0041.
- CV mean score: -0.8201, std: 0.0062.
- CV mean score: -1.3792, std: 0.0113.
- CV mean score: -1.0056, std: 0.0051.
- CV mean score: -1.2725, std: 0.0032.
- CV mean score: -0.7200, std: 0.0030.
- CV mean score: -1.6033, std: 0.0082.

LB score: -1.582

### l1 regression without oof fc

- CV mean score: -0.2321, std: 0.0012.
- CV mean score: -1.4712, std: 0.0041.
- CV mean score: -0.8201, std: 0.0062.
- CV mean score: -1.3792, std: 0.0113.
- CV mean score: -1.0056, std: 0.0051.
- CV mean score: -1.2725, std: 0.0032.
- CV mean score: -0.7200, std: 0.0030.
- CV mean score: -1.6033, std: 0.0082.

LB score: -1.063
