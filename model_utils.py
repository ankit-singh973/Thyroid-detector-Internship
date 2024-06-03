import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully from %s", filepath)
        return df
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def train_extra_trees(X_train, y_train):
    try:
        extra_tree_forest = ExtraTreesClassifier(n_estimators=200, criterion='entropy', max_features=2, max_depth=20)
        extra_tree_forest.fit(X_train, y_train)
        feature_importance = extra_tree_forest.feature_importances_
        feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis=0)
        pd.Series(feature_importance_normalized, index=X_train.columns).plot.bar(color='green', figsize=(12, 6))
        plt.xlabel('Feature Labels')
        plt.ylabel('Feature Importances')
        plt.title('Comparison of different Feature Importances')
        plt.show()
        logging.info("Extra Trees Classifier trained successfully")
        return extra_tree_forest
    except Exception as e:
        logging.error("Error training Extra Trees Classifier: %s", e)
        raise

def drop_unimportant_features(X):
    try:
        cols_to_drop = ['query_on_thyroxine', 'on_antithyroid_medication', 'pregnant', 'thyroid_surgery',
                        'I131_treatment', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'T4U']
        X_dropped = X.drop(columns=cols_to_drop)
        logging.info("Dropped unimportant features: %s", cols_to_drop)
        return X_dropped
    except Exception as e:
        logging.error("Error dropping unimportant features: %s", e)
        raise

def xgboost_model(train_x, train_y):
    try:
        xgb = XGBClassifier()
        param_grid_xgboost = {'tree_method': ['auto'], 'n_estimators': [10, 100, 120], 'booster': ['dart', 'gbtree'],
                              'max_depth': [2, 3], 'alpha': (1e-4, 1), 'colsample_bytree': (.1, .5), 'subsample': (.1, 1)}
        grid = GridSearchCV(xgb, param_grid_xgboost, n_jobs=-1, verbose=3, cv=5)
        grid.fit(train_x, train_y)
        xgb_best = XGBClassifier(**grid.best_params_)
        xgb_best.fit(train_x, train_y)
        logging.info("Best parameters for XGBoost: %s", grid.best_params_)
        return xgb_best
    except Exception as e:
        logging.error("Error training XGBoost model: %s", e)
        raise

def random_forest_model(train_x, train_y):
    try:
        rfc = RandomForestClassifier()
        param_grid = {"n_estimators": [10, 20, 30, 50, 70, 100, 120], "criterion": ['gini', 'entropy'],
                      "max_depth": range(2, 4, 1), "max_features": ['sqrt', 'log2'], "ccp_alpha": (1e-4, 10)}
        grid = GridSearchCV(rfc, param_grid=param_grid, n_jobs=-1, verbose=3, cv=10)
        grid.fit(train_x, train_y)
        rfc_best = RandomForestClassifier(**grid.best_params_)
        rfc_best.fit(train_x, train_y)
        logging.info("Best parameters for Random Forest: %s", grid.best_params_)
        return rfc_best
    except Exception as e:
        logging.error("Error training Random Forest model: %s", e)
        raise

def knn_model(train_x, train_y):
    try:
        knn = KNeighborsClassifier()
        param_grid_knn = {'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 17, 24, 28, 30, 35],
                          'n_neighbors': [4, 5, 8, 10, 11], 'p': [1, 2]}
        grid = GridSearchCV(knn, param_grid_knn, n_jobs=-1, cv=10, verbose=3)
        grid.fit(train_x, train_y)
        knn_best = KNeighborsClassifier(**grid.best_params_)
        knn_best.fit(train_x, train_y)
        logging.info("Best parameters for KNN: %s", grid.best_params_)
        return knn_best
    except Exception as e:
        logging.error("Error training KNN model: %s", e)
        raise

def best_model(train_x, train_y, test_x, test_y):
    try:
        knn = knn_model(train_x, train_y)
        prediction_knn = knn.predict_proba(test_x)
        knn_score = roc_auc_score(test_y, prediction_knn, multi_class='ovr') if len(test_y.unique()) > 1 else accuracy_score(test_y, prediction_knn)
        logging.info('KNN Score: %s', knn_score)

        rfc = random_forest_model(train_x, train_y)
        prediction_rfc = rfc.predict_proba(test_x)
        rfc_score = roc_auc_score(test_y, prediction_rfc, multi_class='ovr') if len(test_y.unique()) > 1 else accuracy_score(test_y, prediction_rfc)
        logging.info('Random Forest Score: %s', rfc_score)

        xgb = xgboost_model(train_x, train_y)
        prediction_xgb = xgb.predict_proba(test_x)
        xgb_score = roc_auc_score(test_y, prediction_xgb, multi_class='ovr') if len(test_y.unique()) > 1 else accuracy_score(test_y, prediction_xgb)
        logging.info('XGBoost Score: %s', xgb_score)

        models_scores = [('KNN', knn, knn_score), ('RandomForest', rfc, rfc_score), ('XGBoost', xgb, xgb_score)]
        best_model_name, best_model, best_model_score = max(models_scores, key=lambda item: item[2])
        logging.info("Best model: %s", best_model_name)
        return best_model_name, best_model
    except Exception as e:
        logging.error("Error determining best model: %s", e)
        raise
