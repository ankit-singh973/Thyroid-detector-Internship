import logging
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from model_utils import (
    load_data, train_extra_trees, drop_unimportant_features,
    best_model
)

# Setting up logging
log_folder = 'training_log'
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, 'model_training.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == "__main__":
    try:
        # Load your data here
        data_path = 'notebook/final_thyroid_1.csv'
        df = load_data(data_path)

        X = df.drop(columns=['Class'])
        y = df.Class

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=70)

        # Train and plot feature importance using Extra Trees
        train_extra_trees(X_train, y_train)

        # Drop unimportant features
        X_train = drop_unimportant_features(X_train)
        X_test = drop_unimportant_features(X_test)

        # Find and train the best model
        model_name, final_model = best_model(X_train, y_train, X_test, y_test)

        # Create the models folder if it doesn't exist
        models_folder = 'models'
        os.makedirs(models_folder, exist_ok=True)

        # Save the best model
        model_path = os.path.join(models_folder, 'final_model.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(final_model, file)
        logging.info("Model training complete. Best model: %s", model_name)
    except Exception as e:
        logging.error("Error in main execution: %s", e)
