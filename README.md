# Credit Card Fraud Detection Project

The Credit Card Fraud Detection project aims to deploy a machine learning model as a Flask API for real-time predictions of fraudulent credit card transactions.

## Project Structure
```
Credit Card Fraud Detection Project/
├── data/
│   └── creditcard.csv # Dataset file containing credit card transaction data
├── notebooks/  #Script to run pipeline code
│   └── credit_card_fraud_detection.ipynb # Jupyter notebook for exploratory analysis (optional)
├── src/
│   ├── preprocessing.py # Python script for data preprocessing
│   ├── modeling.py # Python script for building machine learning models
│   └── evaluation.py # Python script for model evaluation
├── models/
│   └── best_random_forest_model.pkl # Saved Random Forest model
├── report/
│   └── report.pdf # Project report documenting methodology, results, and conclusions
├── README.md # Project README file
├── app.py     # Flask application for deployment
├── requirements.txt # Lists project dependencies (including Flask if used)
└── Dockerfile # Instructions for building a Docker image 
```

## Setup Instructions

1. Clone the repository:

git clone https://github.com/Nerd-coderZero/credit-card-fraud-detection.git


2. Install the required dependencies:

pip install -r requirements.txt


3. Install Docker on your system if not already installed.

## Usage

- Place the dataset file `creditcard.csv` in the `data/` directory.
- Open the Jupyter notebook and run `credit_card_fraud_detection.ipynb` in notebooks folder and follow the instructions to load, preprocess, model, and evaluate the data.
- The trained model will be saved as `best_random_forest_model.pkl` in the `models/` directory.
- The evaluation results will be presented in the notebook and can also be found in the `report/` directory.

## Depoloyment

- Build the Docker image:[ docker build -t credit-card-fraud-detection-app .]
- Run the Docker container:[ docker run -p 5000:5000 credit-card-fraud-detection-app]
- Once the container is running, access the API at http://localhost:5000/predict.

## Testing (Optional)

- Test the API using tools like Postman or curl: [curl -X POST -H "Content-Type: application/json" -d '{"features": [value1, value2, ..., value29]}' http://localhost:5000/predict]

## Conclusion

- The deployment of the Credit Card Fraud Detection API using Docker provides a scalable and efficient solution for real-time fraud detection in credit card transactions.


## Contributors

- [Kushagra Jaiswal](https://github.com/Nerd-coderZero)






