# 🧠 Mental Health Detector

Mental Health Detector is a machine-learning-based Streamlit application that analyzes user-provided text and classifies it into one of seven mental-health-related categories: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal.

## 📌 About the Application

This project uses Natural Language Processing (NLP) and Machine Learning to classify text into different mental-health-related categories.

The application provides an interactive Streamlit dashboard where users can:

- Enter text for prediction
- Get a predicted mental health category
- View confidence scores for all categories
- Explore the dataset
- View model performance
- Visualize data using charts



## 🚀 Features

- 📝 Text-based mental health classification
- 🧠 Seven prediction categories
- 📊 Confidence distribution visualization
- 📈 Exploratory Data Analysis (EDA)
- 📉 Model performance visualization
- 🌐 Interactive Streamlit interface

## 🧩 Prediction Categories

The model classifies text into the following seven categories:

1. Anxiety
2. Bipolar
3. Depression
4. Normal
5. Personality Disorder
6. Stress
7. Suicidal

## 🤖 Machine Learning

The project uses:

- **TF-IDF** for text feature extraction
- **XGBoost** for multi-class classification

### Model Configuration

- TF-IDF maximum features: 5000
- N-gram range: (1, 2)
- Number of classes: 7
- XGBoost estimators: 180
- Maximum tree depth: 6
- Learning rate: 0.08

## 📊 Model Performance

- Training Accuracy: **91.4%**
- Test Accuracy: **77.4%**

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- TF-IDF
- Natural Language Processing (NLP)

## 📁 Project Structure

```text
Mental-Health-Detector/
│
├── data/
│   └── dataset.csv
│
├── Mentalhealth.py
├── README.md
├── requirements.txt
└── .gitattributes
```



### 1. Clone the Repository

```bash
git clone https://github.com/ay6560677-anshul/Mental-Health-Detector.git
```

### 2. Open the Project Folder

```bash
cd Mental-Health-Detector
```

### 3. Install Required Libraries

```bash
pip3 install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run Mentalhealth.py
```


