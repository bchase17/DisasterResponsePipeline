# Disaster Response Pipeline Project

The primary objective of this project is to harness pre-labeled disaster messages for the creation of a real-time disaster response model. This model will enable the categorization of incoming messages during a disaster event, ensuring that each message is directed to the appropriate disaster response agency. Within this endeavor, we've developed a web application tailored to the needs of disaster response workers. The application allows these professionals to input incoming messages and promptly receive classification results. The project leverages a dataset populated with pre-labeled tweets and messages from actual disaster events. The core focus is on developing a Natural Language Processing (NLP) model capable of categorizing messages in real-time.

Key Project Components:

Data Processing: Creation of an ETL pipeline, responsible for data extraction, cleaning, and storage in a SQLite database.
Machine Learning Pipeline: Construction of a robust machine learning pipeline for text message classification across diverse categories.
Real-time Web Application: Deployment of a web application for immediate model results. This application serves as an essential tool for emergency workers, allowing them to input new messages and receive classification results. Additionally, it provides interactive visualizations of the data.
Ultimately, the project is centered around a comprehensive dataset of authentic messages sent during disaster events, for which we've built a cutting-edge machine learning pipeline. This enables the precise routing of messages to the appropriate disaster relief agencies, all facilitated through an intuitive web application interface.

## Table of Contents
1. Setup/Requirements and Instructions for Running 
2. File Descriptions
3. Installation
4. Acknowledgements

## Setup/Requirements and Instructions for Running

### Library Dependencies

- SQL Databases: SQLalchemy
- ML Libraries: Pandas, Numpy, Sciki-Learn, Pickle, NLTK (Natural Language Processing)
- Weh App + Visualizing: Flask, Plotly

### Running

To clean and store the cleaned data in a databse run the below script: 
	data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db

To load the DB and run run the ML pipeline tasked with training and saving the classifier run the below script: 
	models/train_classifier.py 	data/disaster_response_db.db models/classifier.pkl

## File Descriptins

### Folders

app: contains the python script run.py --> this creates the web application that visualizes different metrics about disaster data and allows you to classify specific phrases/words into their most relevant categories through a built in search function

data: contains the python script process_data.py --> this is the Extract-Train-Load code that cleans the data, extracts the relevant features, loads it into a database for later use

models: contains the python script train_classifier.py --> this is tasked with actually training our model by using a machine learning algorithm. The ML loads the data and uses statistcial analysis to train the model through minimizing the margin of error of its predicitons, then it saves the trained model into a .pkl file for later use

## Installation

The following libraries are required to execute this process in full:
- Numpy
- Pandas
- Matplotlib
- Sklearn
- NLTK
- Re
- OS
- Sys
- Pickle
- Sqlalchemy
- Json
- Plotly

## Acknowledgements

- Figure Eight for the dataset to train
- Udacity for their impeccable work on this Data Science nano-degree