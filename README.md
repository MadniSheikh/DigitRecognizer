# ✍️ Handwritten Digit Recognizer

![Status](https://img.shields.io/badge/Status-In%20Progress-blueviolet.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App%20Deployment-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow.svg)

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [Repository Structure](#repository-structure)
- [Connect with Me](#connect-with-me)

---

## <a id="project-overview"></a>📌 Project Overview
This project is an interactive machine learning application that predicts hand-drawn numbers. Built using an Artificial Neural Network (ANN) trained on the classic MNIST dataset, the app takes user input from a digital drawing canvas, preprocesses the image (handling bounding boxes, aspect ratio preservation, and center-of-mass alignment), and passes it to the deep learning model to classify the digit.

By exposing the model through a web interface, anyone can test the neural network's accuracy with their own handwriting in real-time.

### <a id="live-demo"></a>🚀 Live Demo
**Try the interactive web application globally here:** [https://digitrecognizer-cpu.streamlit.app/](https://digitrecognizer-cpu.streamlit.app/)

*(Note: The app may take a few seconds to wake up if it hasn't been used recently.)*

---

## <a id="limitations"></a>⚠️ Limitations
As this is an active, evolving project currently utilizing a standard ANN, there are a few limitations:
* **Single Digits Only:** The model is trained strictly on isolated digits (0-9). 
* **No Multi-Digit Support:** Drawing a "10" or "42" will confuse the model, as it expects one number per canvas.
* **Spatial Sensitivity:** Because ANNs flatten images into 1D arrays, extreme placement or weird drawing angles might trick the model (though custom preprocessing heavily mitigates this).

---

## <a id="future-roadmap"></a>🗺️ Future Roadmap
This project is currently **In Progress**. The next major updates include:
1. **Upgrading to a CNN:** Replacing the current Artificial Neural Network with a Convolutional Neural Network (CNN) to vastly improve spatial awareness and edge detection.
2. **Full Handwriting Recognizer:** Evolving the pipeline beyond single digits to build a comprehensive system capable of recognizing full handwritten words and multi-digit numbers.

---

## <a id="tech-stack"></a>🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras (Sequential ANN)
* **Machine Learning & Scaling:** Scikit-Learn (`StandardScaler`)
* **Computer Vision / Image Processing:** PIL (Pillow), NumPy
* **Frontend/UI:** Streamlit, `streamlit-drawable-canvas`
* **Local GUI Framework:** Tkinter (for local testing)

---

## <a id="how-to-run-locally"></a>💻 How to Run Locally

If you want to clone this repository and experience the code from scratch on your own machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/MadniSheikh/Digit-Recognizer.git
cd Digit-Recognizer
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows use:
venv\Scripts\activate
# On Mac/Linux use:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Choose Your Interface (Web App vs. Local GUI)

**Option A: Run the Global Streamlit Web App**
Launch the modern web browser interface.
```bash
streamlit run app.py
```

**Option B: Run the Local Tkinter GUI**
Experience the digit recognizer locally using the custom-built python desktop window. This is a great way to see the raw Python code working from scratch.
```bash
python digit_gui.py
```

---

## <a id="repository-structure"></a>📂 Repository Structure
```text
├── model/
│   └── ann_model1.h5
├── Notebook/
│   └── mnist-with-ann-98-50.ipynb
├── utils/
│   └── scaling.pkl
├── .gitignore
├── app.py
├── digit_gui.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## <a id="connect-with-me"></a>🤝 Connect with Me

Hi, I'm Madni! I am passionate about bridging the gap between Artificial Intelligence and real-world applications, building end-to-end machine learning systems. Let's connect:

* **LinkedIn:** [www.linkedin.com/in/mohmedmadni-sheikh-](https://www.linkedin.com/in/mohmedmadni-sheikh-)
* **GitHub:** [https://github.com/MadniSheikh](https://github.com/MadniSheikh)
* **Kaggle:** [https://www.kaggle.com/madnishaikh](https://www.kaggle.com/madnishaikh)
