# modeling.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def build_model(model_type, X_train, y_train):
    if model_type == 'best_random_forest':
        # Define the grid of hyperparameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create the grid search object
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_grid=param_grid,
                                   scoring='f1',
                                   cv=3,
                                   n_jobs=-1)
        
        # Perform grid search cross-validation
        grid_search.fit(X_train, y_train)
        
        # Print the best parameters
        print("Best hyperparameters found:", grid_search.best_params_)
        
        # Train the Random Forest model with the best hyperparameters
        model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    else:
        raise ValueError("Invalid model type. Supported types: 'best_random_forest'")
