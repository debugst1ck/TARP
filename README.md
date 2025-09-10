# TARP: Transformers for Antimicrobial Resistance Prediction
This repository contains code and models for predicting antimicrobial resistance using transformer architectures. The models are trained on genomic data to classify bacterial strains based on their resistance profiles.
## Developer's notes
The codebase is structured to facilitate easy experimentation with different transformer architectures and hyperparameters. The main components include data preprocessing, model training, evaluation, and visualization of results.
### Attention Mask
A value of 1 or True indicates that the model should attend to this position. This is for the actual content of the input. A value of 0 or False indicates that the model should not attend to this position, typically because it is padding.