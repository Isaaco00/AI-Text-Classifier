# AI Support Ticket Classifier

A machine-learning application that classifies customer support messages into operational categories and recommends how each ticket should be handled.

**Live Demo:** https://ai-text-classifier-pdlzasggtukd7emqwbeyu.streamlit.app/

## Overview

Support teams receive large numbers of customer messages that need to be assigned to the right department.

This project demonstrates an automated ticket-routing system that uses supervised machine learning to classify incoming messages into four categories:

* Shipping
* Account
* Technical
* General Support

The application also uses model confidence to determine whether a prediction should be automatically routed, reviewed by a human, or clarified with the customer.

## How It Works

```text
Customer Message
       ↓
Text Preprocessing
       ↓
TF-IDF Feature Extraction
       ↓
Logistic Regression Classifier
       ↓
Predicted Category + Confidence
       ↓
Routing Decision
       ↓
Automatic Route / Human Review / Clarification
```

## Machine Learning Approach

The classifier uses a scikit-learn pipeline consisting of:

* **TF-IDF Vectorization** for converting text into numerical features
* **Logistic Regression** for multi-class classification
* **Train/test split** with stratification
* **Classification metrics** including accuracy, precision, recall, and F1-score

The training dataset contains **236 labeled support messages**, evenly distributed across the four categories.

## Model Performance

The current model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 98.31% |
| Precision | 98.42% |
| Recall    | 98.31% |
| F1 Score  | 98.31% |

Test set:

* Training samples: 177
* Test samples: 59
* Categories: 4

### Classification Performance

| Category        | Precision | Recall |   F1 |
| --------------- | --------: | -----: | ---: |
| Account         |      1.00 |   1.00 | 1.00 |
| General Support |      0.93 |   1.00 | 0.97 |
| Shipping        |      1.00 |   0.93 | 0.97 |
| Technical       |      1.00 |   1.00 | 1.00 |

## Confidence-Based Routing

The application does not blindly trust every prediction.

Instead, model confidence determines the recommended operational action:

* **80%+** → Automatic routing
* **50–79%** → Human review
* **Below 50%** → Request clarification

This creates a safer workflow where uncertain predictions can be handled by people instead of being automatically routed.

## Application Features

* Real-time ticket classification
* Four support categories
* Model confidence scoring
* Confidence-based routing decisions
* Model performance dashboard
* Quick test examples
* Prediction history
* Human-review fallback
* Streamlit web interface
* Automatic model creation when the saved model is unavailable

## Project Structure

```text
AI-Text-Classifier/
│
├── app/
│   └── app.py
│
├── data/
│   ├── create_dataset.py
│   └── training_data.csv
│
├── models/
│   └── text_classifier.joblib
│
├── src/
│   ├── classifier.py
│   ├── decision.py
│   └── preprocessor.py
│
├── tests/
│   └── test_classifier.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> The trained `.joblib` model is intentionally excluded from Git tracking. The application can recreate the model from the training dataset when necessary.

## Running Locally

Clone the repository:

```bash
git clone https://github.com/Isaaco00/AI-Text-Classifier.git
cd AI-Text-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

## Running Tests

Run the classifier validation:

```bash
python tests/test_classifier.py
```

Validate Python syntax:

```bash
python -m py_compile app/app.py
python -m py_compile src/classifier.py
python -m py_compile src/decision.py
python -m py_compile src/preprocessor.py
python -m py_compile data/create_dataset.py
```

## Technologies

* Python
* Pandas
* Scikit-learn
* Streamlit
* Joblib
* Git
* GitHub

## Limitations

This is a portfolio-scale machine-learning application rather than a production customer-support system.

The dataset is relatively small and synthetic, so real-world performance would require a much larger collection of genuine, anonymized support tickets.

Model confidence should also not be interpreted as guaranteed correctness.

## Future Improvements

Potential future improvements include:

* Larger real-world training datasets
* More support categories
* Better probability calibration
* Human feedback collection
* Persistent prediction storage
* API deployment
* Authentication
* Monitoring and model drift detection
* Automated retraining
* Integration with ticketing platforms

## Author

**Gbadebo Oluwaseunfunmi Isaac**

Computer Science student building practical machine-learning and software projects.

GitHub: https://github.com/Isaaco00
