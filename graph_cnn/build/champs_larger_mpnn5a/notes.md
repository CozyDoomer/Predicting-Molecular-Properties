## notes for the mpnn

- features converted to graph pickle files in data

- graph files are read in __getitem__ in dataset.py (ChampsDataset class)

- ChampsDataset class is used in train.py, run_train to extract pickle files 

- null_collate is used when creating the ChampsDataset dataloader to combine the variable size graph features

- these features are passed to the model.py forward function to a lstm mechanism

## TODO

- try oof fc values for train and test

- find better angle features 
    - there are calculations for cosine and dehidral seperatly
    - cosine would benefit 2J couplings and dehidral 3J

- find normalization for lstm