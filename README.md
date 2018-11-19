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
Eventhough this repository comes with all files needed to run the trained text classification model in a web app already, you're free to choose if you want to train a new model on the existing or a new message data set. Should you only be interested in deploying the web app for your purposes you can jump straight ahead to [__Deployment__]().

### Pre-Processing

Data

### Training

Class imbalance

### Deployment

To cloud service

## Acknowledgements
