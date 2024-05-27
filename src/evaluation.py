# evaluation.py

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importances
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    columns = X_test.columns
    
    # Plot feature importances
    plt.figure(figsize=(15, 8))
    plt.title("Feature Importances")
    plt.bar(range(X_test.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X_test.shape[1]), columns[indices], rotation=90)
    plt.xlim([-1, X_test.shape[1]])
    plt.show()
