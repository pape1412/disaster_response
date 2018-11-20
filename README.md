# Disaster Response
Exploring ETL and ML pipelines for text classification.

## Intro
Whenever natural disaster occur it's important to deliver aid and assistance to those who need it the most. Knowing quickly where the ones affected are and what they require can be key for disaster response teams in order to save human lifes. But how to find the right information out of an endless amount of messages such as on the internet or the news?

This project tries to explore how __machine learning__ can help organizations to make __better aid and assistance related decisions__, especially during and after natural disasters.

## Installation
Despite standard libraries that come with the Anaconda distribution of Python you'll need ```sklearn```, ```nltk``` and ```sqlalchemy``` for both model training and web app deployment. Also, you'll need ```flask``` and ```plotly``` in order to successfully run the app. For further information on packages and versions please see the ```requirements.txt``` file in the repository.

## Files
Files within this repository a separated into __three main groups__, namely:
- ```app``` includes all files related to the web app's backend and frontend architecture
- ```data``` includes all files related to the data used in the project
- ```models``` includes all files related to model training as well as the trained model

Despite that you can use __the following tree for orientation__ and additional explanation of files:
```
- app
| - template
| |- master.html            # Main page of web app
| |- go.html                # Classification result page of web app
|- run.py                   # Flask file that runs app

- data
|- disaster_categories.csv  # Data to process 
|- disaster_messages.csv    # Data to process
|- process_data.py          # Data pre-processing
|- disaster_response.db     # Database to save pre-processed data to

- models
|- train_classifier.py      # Model training
|- classifier.pkl           # Saved model
|- custom_transformer.py    # Custom transformer for sklearn pipeline

- README.md
- requirements.txt
```

## Usage
Eventhough this repository comes with all files needed to run the trained text classification model in a web app already, you're free to choose if you want to train a new model on the existing or a new message data set. Should you only be interested in deploying the web app for your purposes you can jump straight ahead to [Deployment](https://github.com/pape1412/disaster_response/blob/master/README.md#deployment).

### ETL Pipeline
#### Data
The data that's being used in this project comes from __two disaster messages data sets__ provided by [Figure Eight](https://www.figure-eight.com). Both of these data sets require additional pre-procsessing steps in order to be used for training a model.

#### Pre-Processing
The pre-processing is done within a __small ETL (Extract, Transform, Load) pipeline__, that loads the ```disaster_messages.csv``` and ```disaster_categories.csv``` data sets, merges them together, cleans the data (e.g. creation of category labels, removal of duplicates, ...) and then stores a new data set in a SQLite database.

Should you wish to re-run these pre-processing steps with a new or the existing data set you can do so by __executing the following line__ from the main folder:
```
$ python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
The command takes __three positional arguments__, namely both the paths to the messages and categories data sets as well as the path to a new SQLite database.

### ML Pipeline
#### Training
The machine learning pipeline includes all steps necessary to successfully train a __multi-label disaster response classification model__. After loading the pre-processed data set from the SQLite database, it splits the data and sets up a text processing and machine learning pipeline with sklearn's ```Pipeline```, ```FeatureUnion``` and ```MultiOutputClassifier``` modules. It then runs a randomized grid search with cross-validation of a medium sized parameter space for 10 iterations (trains only 10 different models instead of running a full search over the cartesian grid of parameters). Last but not least the pipeline finishes with an output of model evaluation metrics and then saves the best performing model to a ```.pkl``` file.

Eventhough you find a trained classification model Again, should you wish to re-train the model you can
```
$ python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
#### Evaluation
Class imbalance

### Flask Web App
Finally, you can use the pre-processed data as well as the trained disaster response classification model in a simple web app.

```
$ python3 app/run.py
```

Visit web app at ```http://0.0.0.0:3001/``` in local browser.

Screenshots

Future Ideas:
- Deployment to cloud service

## Acknowledgements
