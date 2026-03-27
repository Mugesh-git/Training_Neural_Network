# 🌌 Planet Growth Stage Classification using Neural Networks

A simple machine learning project that uses a neural network to classify the growth stage of planets based on simulated astronomical features.

---

## 📌 Overview

This project generates synthetic planetary data and trains a neural network model to classify planets into different growth stages.

The model predicts one of three stages:

* Stage 0
* Stage 1
* Stage 2

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras

---

## ⚙️ Features Used

The model is trained on the following features:

* Planet Size (Earth radii)
* Orbital Period (years)
* Surface Temperature (Kelvin)
* Atmospheric Composition (normalized factor)

---

## 🧠 Model Architecture

* Input Layer → 4 features
* Hidden Layer → 16 neurons (ReLU)
* Hidden Layer → 8 neurons (ReLU)
* Output Layer → 3 neurons (Softmax)

---

## 🚀 Workflow

1. Generate synthetic dataset
2. Encode labels using one-hot encoding
3. Split data into training and testing sets
4. Apply feature scaling using StandardScaler
5. Train a neural network model
6. Evaluate performance on test data
7. Predict and compare results

---

## ▶️ How to Run

Install dependencies:

```bash id="dep1"
pip install numpy pandas scikit-learn tensorflow
```

Run the script:

```bash id="run1"
python your_script_name.py
```

---

## 📊 Output

* Training accuracy and validation accuracy
* Test accuracy
* Predicted vs actual class labels

Example:

```id="out1"
Test Accuracy: 0.85
Predicted classes: [...]
True classes: [...]
```

---

## ⚠️ Limitations

* Uses randomly generated (synthetic) data
* No real-world dataset validation
* Model performance does not reflect real astronomical scenarios
* No hyperparameter tuning

---

## 🔧 Future Improvements

* Use real exoplanet datasets (e.g., NASA data)
* Perform hyperparameter tuning
* Add model evaluation metrics (precision, recall, confusion matrix)
* Visualize training performance
* Deploy as a web app

---

## ⚠️ Disclaimer

This project is for educational purposes only and uses simulated data.

---

## 👤 Author

Mugesh
