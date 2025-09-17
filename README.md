# TARP: Transformers for Antimicrobial Resistance Prediction 
This repository is a suite of tools and models designed to predict antimicrobial resistance (AMR) using transformer-based architectures. The project uses state-of-the-art techniques in natural language processing (NLP) to analyze genetic sequences and predict resistance profiles.
## Developer's notes
The codebase is structured to facilitate easy experimentation with different transformer architectures and hyperparameters. The main components include data preprocessing, model training, evaluation, and visualization of results.
### Attention Mask
A value of 1 or True indicates that the model should attend to this position. This is for the actual content of the input. A value of 0 or False indicates that the model should not attend to this position, typically because it is padding.

### Class Weights
Class weights are calculated to address class imbalance in the dataset. The weights are inversely proportional to the frequency of each class, ensuring that the model pays more attention to minority classes during training.

$$
\text{weight}_i = \frac{N}{C \cdot n_i}
$$