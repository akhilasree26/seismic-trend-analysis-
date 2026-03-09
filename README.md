# 🌍 Seismic Trend Analysis using Deep Learning

This project analyzes earthquake patterns using hybrid deep learning architectures to classify strong earthquakes.  
A comparison is performed between **CNN-LSTM** and **CNN-GRU** models to evaluate their performance on seismic datasets.

📄 **Research Paper Published**

This research work compares deep learning models for earthquake magnitude classification and seismic pattern analysis.


# 📊 Project Overview

Earthquake prediction and seismic analysis are important for disaster management.  
This project applies **hybrid deep learning models** to identify patterns in historical earthquake data and classify high magnitude earthquakes.

Two architectures were implemented:

• **CNN + LSTM Hybrid Model**  
• **CNN + GRU Hybrid Model**

CNN layers extract **spatial features** from seismic parameters while LSTM/GRU layers capture **temporal dependencies**.



# 🧠 Model Architecture
Input Features
↓
CNN Layers (Feature Extraction)
↓
LSTM / GRU Layers (Temporal Pattern Learning)
↓
Dense Layers
↓
Binary Classification (Magnitude ≥ 6)




# 📂 Dataset

Dataset used:

Earthquake Database (1965–2016)

Features used:

- Latitude
- Longitude
- Depth
- Azimuthal Gap
- Horizontal Distance
- Horizontal Error
- Root Mean Square

Target:

Magnitude ≥ 6 → Strong Earthquake



# ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn



# 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- RMSE
- R² Score
- Confusion Matrix


# 📊 Model Comparison

| Model | Strength |
|------|------|
| CNN + LSTM | Good long-term temporal learning |
| CNN + GRU | Faster training and better stability |

Results show that **CNN-LSTM performed best. Because LSTM captures long-term temporal dependencies more effectively.**



# 📉 Visualizations

The project includes:

- Training vs Validation Accuracy
- Training vs Validation Loss
- Confusion Matrix
- Predicted Probability Distribution



# 📁 Project Structure
Seismic-Trend-Analysis

├── cnn-lstm-model.py
├── cnn-gru-model.py
├── requirements.txt
├── research-paper.pdf
└── README.md


---

# 👩‍💻 Authors

- Akhila Sree Menda
- Suma Dasari  
- Mohitha Sree Boggavarapu  
- Dr. Rajkumar Yesuraj  
School of Computer Science and Engineering  
Vellore Institute of Technology, Amaravati

---

# ⭐ If you found this project useful

Please consider **starring ⭐ the repository** on GitHub.

