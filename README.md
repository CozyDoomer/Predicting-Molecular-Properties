# Predicting-Molecular-Properties

- https://www.kaggle.com/c/champs-scalar-coupling

## Results so far 

## Message Passing NN

### dataset setting
```
batch_size = 10

train_dataset : 
	mode   = train
	split  = train_split_by_mol.80003.npy
	csv    = train
	len    = 80003

valid_dataset : 
	mode   = train
	split  = valid_split_by_mol.5000.npy
	csv    = train
	len    = 5000

## net setting

<class 'model.LargerNet'>

optimizer
  Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0002
    max_lr: 0.001
    weight_decay: 0
)

scheduler
  NullScheduler
lr=0.00010 

 batch_size =10,  iter_accum=1
```

| lr      | iter   | epoch | 1JHC    | 2JHC    | 3JHC    | 1JHN    | 2JHN    | 3JHN    | 2JHH    | 3JHH   | loss   | mae | log_mae | loss | time |
|---------|--------|-------|---------|---------|---------|---------|---------|---------|---------|--------|--------|--------|--------------|---------|--------|
| 0.00010 | 325.0* | 39.9  | -1.046, | -1.887, | -1.591, | -1.073, | -2.081, | -2.238, | -2.048, | -1.899 | -1.644 | 0.19 | -1.73 | -1.995 | about 15 hours |

### public score: -1.717

## Lightgbm, meta-feature kaggle notebook as starting point

### scores for l2 regression

with lambda_l1 = 0.1 and lambda_l2 = 0.3

LB score: -1.584

### best mae/l1 regression score

lambda_l1 = 0.1, lambda_l2 = 0.3, factor = np.tanh(variance/20)+1

LB score: -1.611

### l1 regression with different params

lambda_l1 = 0.4, lambda_l2 = 0.2, factor = np.tanh(variance/30)+1

LB score: -1.582

### l1 regression without oof fc

LB score: -1.063
