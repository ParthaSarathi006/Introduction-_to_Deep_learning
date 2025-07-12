# 🧠 Introduction to Deep Learning with TensorFlow

## 📌 Project Overview

This project introduces the basics of **Deep Learning** using the **Breast Cancer Wisconsin dataset**.  
It demonstrates a complete end-to-end pipeline: data preparation, model building with **TensorFlow/Keras**, training, and evaluation.

---

## 📁 Dataset Information

- 🧬 Dataset: Breast Cancer Wisconsin Diagnostic Dataset (from `sklearn.datasets`)  
- **🎯 Target variable**: `Label` (1 = Malignant, 0 = Benign)  
- **📊 Features**: 30 numerical input features such as:
  - `mean radius`, `mean texture`, `mean smoothness`, etc.

---

## 🧹 Data Preprocessing

1. ✅ Loaded dataset using `sklearn.datasets.load_breast_cancer()`  
2. 🗃 Converted to pandas DataFrame and added `Label` column  
3. ❌ Checked and confirmed there were **no missing values**  
4. 📤 Split into `x` (features) and `y` (labels)  
5. 🔀 Performed `train_test_split` (80% train, 20% test)  
6. ⚖️ Scaled features using `StandardScaler` for better model performance

---

## 🧠 Model Architecture

- Built using **TensorFlow Keras Sequential API**  
- Architecture:

  ```python
  keras.Sequential([
      keras.layers.Flatten(input_shape=(30,)),
      keras.layers.Dense(20, activation='relu'),
      keras.layers.Dense(2, activation='sigmoid')
  ])
  ```

- 🧪 Compilation:
  - Optimizer: `adam`  
  - Loss: `sparse_categorical_crossentropy`  
  - Metrics: `accuracy`  

---

## 🏋️ Model Training

- Trained for **50 epochs** using `model.fit()`  
- Used **validation_split=0.1** to evaluate on unseen training data  
- Training and validation curves plotted using `matplotlib`

---

## 📈 Evaluation & Prediction

- 🧠 Model accuracy and loss visualized with line plots  
- 🧾 Made predictions using `model.predict()`  
- 🔢 Converted probabilities to class labels using:
  ```python
  pred = np.argmax(ypred, axis=1)
  ```

---

## 📌 Key Observations

- 🏥 Neural networks can effectively classify health-related data  
- ✅ Feature scaling is essential for better convergence  
- 📊 Accuracy curves help identify overfitting or underfitting  
- 🎯 `argmax` is used to interpret model output into predicted class

---

## 🧰 Libraries Used

- `tensorflow` 🤖  
- `numpy` 🔢  
- `pandas` 🐼  
- `sklearn` 🧪  
- `matplotlib.pyplot` 📊  

---

## 🚀 Future Improvements

- Add more hidden layers to test deeper networks  
- Apply techniques like Dropout or Batch Normalization  
- Evaluate using confusion matrix, precision, recall, F1-score  
- Build a Streamlit app for real-time predictions  

---

## 🙏 Acknowledgments

- Dataset from `sklearn.datasets` (UCI Machine Learning Repository)  
- Inspired by introductory Deep Learning workflows

---

## 📜 License

This project is licensed under the **MIT License** ✅
