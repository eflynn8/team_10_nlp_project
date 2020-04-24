# team_10_nlp_project

To generate the results from the Naive Bayes classifier,run the following command from the Models directory: python3 main.py

All files to run Jupyter notebooks can be found in Upload_Files
Results for the models described in the paper can be found in model_interpretation.ipynb. The first model is the MLP trained on continuous features. The files that need to be uploaded to run this notebook are included in this repo and the file location and name is given in the notebook. Following the results of the model on train and test data, the interpretability section includes construction of confusion matrices and heatmap visualizations.
The second model is the LSTM with word embeddings. The necessary files are also listed in that notebook.

The MLP and LSTM can be accessed through feature_model_genre.ipynb and genre_keras_classifier.ipynb, respectively. 
feature_model_genre.ipynb includes the code needed to train the MLP model. This requires several file uploads; the instructions for each file upload are given in the notebook. The model may be trained on either discrete or continuous feature data. Both feature files are included in this repo.
genre_keras_classifier.ipynb includes the code needed to train the LSTM model. The files necessary are specified in the notebook.
