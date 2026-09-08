# 📰 Fake News Detector

An NLP-based machine learning application that classifies news articles as **REAL** or **FAKE**.

The project explores multiple machine learning and deep learning approaches for fake news classification, including **Logistic Regression, Naive Bayes, Random Forest, and LSTM**. A Flask web application provides an interactive interface where users can submit news articles and receive a prediction with a confidence score.

---

## 📌 Project Overview

The spread of misleading and fabricated information through online platforms makes automated news classification an important Natural Language Processing (NLP) problem.

This project uses the **ISOT Fake News Dataset** to train models that learn linguistic and statistical patterns associated with fake and real news articles.

The final application allows a user to:

- Enter a news article
- Select a machine learning or deep learning model
- Process and clean the text
- Generate a prediction
- Display whether the article is classified as **REAL** or **FAKE**
- Display the model's confidence score

> **Important:** This application is a machine-learning classifier, not a fact-checking system. Its predictions are based on patterns learned from the training dataset and should not be treated as proof that a news article is factually true or false.

---

## 🚀 Features

- 📰 REAL / FAKE news classification
- 🧹 NLP-based text preprocessing
- 🔤 Tokenization
- 🛑 Stopword removal
- 🌱 Lemmatization
- 📊 TF-IDF feature extraction
- 🌲 Random Forest classification
- 🧠 LSTM deep learning model
- 🤖 Multiple model comparison
- 📈 Confusion matrix and ROC curve evaluation
- 🎯 Prediction confidence score
- 🌐 Flask-based web interface

---

## 🧠 Machine Learning Approach

The overall workflow of the project is:

```text
                    News Article
                         │
                         ▼
              Text Preprocessing
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Cleaning      Tokenization   Lemmatization
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Feature Extraction
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           TF-IDF                Token Sequences
              │                     │
              ▼                     ▼
       Machine Learning            LSTM
              │                     │
              └──────────┬──────────┘
                         ▼
                  REAL / FAKE
                         │
                         ▼
                 Confidence Score
```

---

## 🔤 NLP Preprocessing

Before being passed to the models, the news article undergoes several preprocessing steps.

### 1. HTML Removal

HTML tags are removed from the text.

### 2. URL Removal

URLs are removed because they generally do not provide useful information for the classification task.

### 3. Special Character Removal

Non-alphabetic characters are removed to simplify the text representation.

### 4. Lowercasing

All text is converted to lowercase so that words such as:

```text
Government
government
GOVERNMENT
```

are treated consistently.

### 5. Tokenization

The text is divided into individual words/tokens.

### 6. Stopword Removal

Common English words that generally provide limited classification value are removed using NLTK stopwords.

Examples include:

```text
the
is
a
an
of
to
```

### 7. Lemmatization

Words are reduced toward their base form using WordNet lemmatization.

This helps reduce unnecessary variation in the vocabulary.

---

## 🤖 Machine Learning & Deep Learning Models

The project experiments with both traditional machine learning and deep learning approaches.

### 1. Logistic Regression

Logistic Regression is used as a traditional baseline classification model.

It works well with high-dimensional sparse text representations such as TF-IDF.

**Accuracy: 99.12%**

---

### 2. Naive Bayes

Naive Bayes is another commonly used algorithm for text classification.

It provides a strong baseline for NLP problems because of its simplicity and effectiveness with word-frequency-based features.

**Accuracy: 94.48%**

---

### 3. Random Forest

Random Forest is an ensemble machine learning algorithm consisting of multiple decision trees.

For this project, the news articles are transformed into TF-IDF features before being passed to the Random Forest classifier.

**Accuracy: 99.84%**

Random Forest is used as the **primary deployed model** in the Flask application.

---

### 4. LSTM

A Long Short-Term Memory (LSTM) neural network is used to explore a deep learning approach.

Unlike the TF-IDF-based models, the LSTM works with sequences of tokens and is designed to capture patterns across word sequences.

The LSTM is also available through the Flask interface for comparison with the Random Forest model.

---

## 📊 Model Performance

The models achieved the following accuracy during evaluation:

| Model | Accuracy |
|---|---:|
| Logistic Regression | **99.12%** |
| Naive Bayes | **94.48%** |
| Random Forest | **99.84%** |
| LSTM | Evaluated separately |

### 🏆 Best Performing Model

Based on the recorded evaluation results, **Random Forest achieved the highest reported accuracy at 99.84%**.

However, high performance on the project dataset does not guarantee the same performance on completely new or modern real-world news articles.

---

## 📈 Model Evaluation

The project includes several evaluation visualizations.

### Confusion Matrix

The confusion matrix shows the number of:

- Correct REAL predictions
- Correct FAKE predictions
- REAL articles classified as FAKE
- FAKE articles classified as REAL

### ROC Curve

The ROC curve is used to evaluate classification performance across different probability thresholds.

### LSTM Evaluation

The LSTM model is evaluated separately from the traditional machine learning models.

Evaluation plots are included in the `notebooks/` directory.

---

## 📚 Dataset

This project uses the **ISOT Fake News Dataset**.

The dataset contains two CSV files:

```text
Fake.csv
True.csv
```

The project contains approximately **44,900 news articles** in total.

The two datasets represent:

- Fake news articles
- Real news articles

The dataset is used for training and evaluating the classification models.

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- Naive Bayes
- Random Forest

### Deep Learning

- TensorFlow
- Keras
- LSTM

### Natural Language Processing

- NLTK
- TF-IDF
- Tokenization
- Stopword Removal
- Lemmatization

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Web Application

- Flask
- HTML
- CSS

### Model Serialization

- Joblib
- Keras model format

---

## 📁 Project Structure

```text
fake-news-detector/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
│   ├── model_training.ipynb
│   ├── confusion_matrix.png
│   ├── confusion_matrix_lstm.png
│   └── roc_curve.png
│
├── saved_models/
│   ├── lstm_model.keras
│   ├── rf_model.pkl
│   ├── tfidf.pkl
│   └── tokenizer.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The trained model files inside `saved_models/` are excluded from normal Git tracking because some of these files are large. The repository therefore contains the training code, notebook, dataset, preprocessing logic, and application code, while trained model artifacts are handled separately.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zedf04/fake-news-detector.git
```

Move into the project directory:

```bash
cd fake-news-detector
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The application will start on:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

---

## 🖥️ Using the Application

### Step 1

Enter a news article into the text box.

### Step 2

Select a model:

```text
Random Forest
LSTM
```

### Step 3

Click **Check News**.

### Step 4

The application displays:

```text
Prediction: REAL / FAKE
Confidence: XX%
Model Used: Random Forest / LSTM
```

---

## 🧪 Example Prediction

### Input

```text
The government announced a new economic policy
designed to support businesses and increase investment.
Officials said the policy would focus on infrastructure
and economic growth.
```

### Example Output

```text
Prediction: FAKE
Confidence: 54%
Model Used: Random Forest
```

> The output above is an example of a model prediction. The confidence score represents the model's confidence in its classification and should not be interpreted as factual certainty.

---

## 🔬 Model Training

The complete model development workflow is available in:

```text
notebooks/model_training.ipynb
```

The notebook covers the major stages of the project, including:

1. Dataset loading
2. Data exploration
3. Text preprocessing
4. Feature extraction
5. Model training
6. Model evaluation
7. Performance comparison
8. Visualization
9. Model preparation for deployment

---

## 🔄 Deployment Architecture

The Flask application connects the trained models with the web interface.

```text
                    User
                      │
                      ▼
                Flask Web App
                      │
                      ▼
              Input News Article
                      │
                      ▼
              Text Preprocessing
                      │
             ┌────────┴────────┐
             ▼                 ▼
           TF-IDF           Tokenizer
             │                 │
             ▼                 ▼
       Random Forest          LSTM
             │                 │
             └────────┬────────┘
                      ▼
                 Prediction
                      │
                      ▼
              Result + Confidence
                      │
                      ▼
                    Web UI
```

---

## ⚠️ Limitations

Although the models achieve strong performance on the project dataset, the system has several limitations.

### Dataset Dependence

The models learn patterns from the ISOT dataset. News articles with substantially different writing styles may produce less reliable predictions.

### Not a Fact Checker

The system does not verify claims against external sources.

For example, if an article contains a false statement written in a style similar to real news, the model may still classify it as REAL.

### Real-World Generalization

Performance on modern news articles may differ from the reported dataset accuracy.

### Model Confidence

A confidence score represents the model's probability estimate. It should not be interpreted as the probability that the article is objectively true or false.

---

## 🚀 Future Improvements

Possible improvements include:

- [ ] Integrate BERT or other transformer-based models
- [ ] Improve generalization to modern news articles
- [ ] Add external fact-checking sources
- [ ] Add source credibility analysis
- [ ] Add explainable AI for predictions
- [ ] Improve handling of very short articles
- [ ] Add more diverse training datasets
- [ ] Deploy the application online
- [ ] Add an API endpoint for programmatic predictions
- [ ] Add automated model monitoring
- [ ] Compare additional NLP architectures

---

## 🎯 Key Learning Outcomes

Through this project, the following concepts were explored:

- Natural Language Processing
- Text preprocessing
- Feature engineering
- TF-IDF vectorization
- Supervised machine learning
- Ensemble learning
- Deep learning
- LSTM networks
- Model evaluation
- Classification metrics
- Model serialization
- Flask application development
- Machine learning model deployment

---

## 👨‍💻 Author

**Sagnik Ghosh**

Computer Science Student  
AI/ML | Data Science | Python | Java

---

## ⭐ Project Purpose

This project was built as a practical exploration of **NLP, machine learning, deep learning, and model deployment**, with the goal of understanding the complete workflow from raw text data to a usable machine learning application.

The project demonstrates an end-to-end machine learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Evaluation
 ↓
Model Serialization
 ↓
Flask Deployment
 ↓
User Prediction
```