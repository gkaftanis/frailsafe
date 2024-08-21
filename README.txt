#README

## Author
Georgios Kaftanis

## Introduction
This README file provides an overview of the code associated with my diploma project. The code is designed to accomplish specific tasks related to my diploma project, and this document will help you understand its purpose, how to use it, and any additional information you may need to work with it.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)

## Project-description
This code is oriented towards the processing of motion and behavior data, as well as the testing of models for predicting the vulnerability of the elderly.

## Installation
To use this code, follow these installation steps:

1. Setup Database: Run database_setup.py file:
python3 database_setup.py --root_password ROOT_PASSWORD --arango_nodes http://localhost:PORT

2. Install Dependencies
pip3 install tsfresh pandas numpy tensorflow multiprocessing scikit-learn imblearn kerastuner xgboost asyncio

3. Modify settings.py file

## Usage
1. Start database container

2. In order to write .mat files to database's collection, move .mat files to data/wwsx_folder_path folder and run:
python3 run_module.py participants_source/plugins/wwsxParticipantsTracker.py 

3. In order to create TimeseriesFeaturesEntry document entries, modify async_main function and FeatureExtractor parameters and run:  
python3 run_module.py models_manager/feature_extractor.py

4. To test models modify parameters of evaluate_models_with_features method of NetworkManager object in async_main function and run:
python3 run_module.py models_manager/network_manager.py
