# ds1003finalproject
### Authors:
Matthew Dunn, Rafael Garcia Cano Da Costa, and Benjamin Jakubowski
### Project Description:
This repository is for a final course project for NYU DS1003: Machine Learning and Computational Statistics. The objective of the project was to develop a predictive model to predict the outcome of US Immigration Court decisions on asylum applications. This repository contains code for
- Data cleaning and merging
- Model training, evaluation, and visualization.

## Usage
First, note data cleaning and merging scripts are described in a separate README (in the cleaning_data subdirectory). Note the data cleaning code is not fully modular (TODO).

Model training, evaluation, and visualization is more modular. The modeling module encapsulates generic functionality for model training, evaluation, and visualization. Then, model families are encapsulated in separate modules, which call functions from the generic modeling module. Note this structure was used to allow each author to independently optimize hyperparameters using the same grid search framework.

To train each module, use the bash command

```
python -m <model_family_module>
```

Note executing a model family module will:
- Save a learning curve showing grid search over a relevant hyperparameter
- Dump a pickled sklearn GridSearchCV object (to save results, since GridSearchCV training can take many hours to run)

