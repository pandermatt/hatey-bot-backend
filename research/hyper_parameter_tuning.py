import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from core.model_trainer import generate_and_train_model

if __name__ == '__main__':
    random_forest = RandomForestClassifier()

    # use the same train data as the model
    ensemble_model = generate_and_train_model(base_classifier=random_forest)
    X = ensemble_model.X_train
    y = ensemble_model.y_train

    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    randomize_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid, n_iter=100, cv=5,
                                          verbose=2, random_state=42, n_jobs=-1)
    randomize_search.fit(X, y)

    print(randomize_search.best_params_)
    print(randomize_search.best_score_)
    print(randomize_search.best_estimator_)
    print(randomize_search.cv_results_)
