
# Naive Bayes Classifier from Scratch

This project demonstrates the implementation of a **Gaussian Naive Bayes classifier** using **Python** and **NumPy**, without the use of any external machine learning libraries such as Scikit-learn. It is intended as a learning exercise to understand the inner workings of one of the most fundamental probabilistic classifiers in machine learning.

---

## 📌 Features

- ✅ Manual computation of class-wise **mean**, **standard deviation**, and **prior probabilities**
- ✅ Likelihood estimation using the **Gaussian (normal) distribution**
- ✅ Prediction by computing **posterior probabilities**
- ✅ Fully functional **from-scratch implementation**
- ✅ Includes a **test case** with synthetic data for demonstration and validation

---

## 🛠️ Technologies Used

- [Python 3](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [Jupyter Notebook](https://jupyter.org/) for interactive development

---

## 🚀 How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/naive-bayes-classifier.git
   cd naive-bayes-classifier
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Then open the `.ipynb` file (e.g. `naive_bayes_classifier.ipynb`).

3. **Run the cells** in order to:
   - Import libraries
   - Define the `NB_Classifier` class
   - Fit the model with training data
   - Predict and evaluate the results

---

## 🔍 Code Structure

### `NB_Classifier` Class

```python
class NB_Classifier:
    def fit(self, X, y):      # Training the model
    def predict(self, X):     # Making predictions
    def get_likelihood(self, x):  # Compute likelihoods for a sample
    def gaussian(self, x, mean, std):  # Gaussian probability formula
```

Each method is modular and serves a specific purpose to maintain clarity and reusability.

---

## 🧪 Example: Classification on Synthetic Data

```python
import numpy as np

X = np.array([[1, 2], [1, 4], [2, 3], [6, 8], [7, 9], [8, 8]])
y = np.array([0, 0, 0, 1, 1, 1])

model = NB_Classifier()
model.fit(X, y)
predictions = model.predict(X)

print(predictions)
```

**Expected Output:**
```
[0 0 0 1 1 1]
```

---

## 📈 Mathematical Background

This implementation is based on **Bayes' Theorem**:

\[
P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
\]

We use the Gaussian distribution for continuous features:

\[
P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right)
\]

Where:
- \(\mu\) = mean of the feature for class \(y\)
- \(\sigma\) = standard deviation of the feature for class \(y\)

---

## 🧠 What You’ll Learn

- How generative classification models work
- How to handle class priors and feature likelihoods
- How to build a robust machine learning model from first principles
- Debugging and testing classifiers manually

---

## 📂 Project Structure

```
naive-bayes-classifier/
│
├── naive_bayes_classifier.ipynb  # Main notebook with implementation & testing
├── README.md                     # Project documentation
└── requirements.txt              # (Optional) Environment dependencies
```

---

## ✨ Extensions You Can Try

- 🔄 Add support for **multiclass datasets**
- 📊 Visualize **decision boundaries**
- 📈 Apply to a **real-world dataset** (e.g., Iris)
- 🤖 Compare with Scikit-learn's implementation

---

## 📚 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributions

Contributions are welcome! If you’d like to:
- Fix bugs
- Add new features
- Improve documentation

Please feel free to open a Pull Request or submit an Issue.
