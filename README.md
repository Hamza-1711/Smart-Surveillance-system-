# Smart-Surveillance-system-
The aim of this project is to design a web application to detect fire and smoke using the modern machine learning techniques like CNNs and RNNs. 

A full-stack deep learning project for detecting **fire** and **smoke** in images and live webcam feed.
---

## 📁 Project Structure
├── Data pre-processing and Model training # Code notebooks/scripts for training models
├── Dataset # Dataset used for training (not included)
├── Models # Saved trained models (RNN .h5 and YOLO .pt)
├── Plots # Visualizations of model training and evaluation
├── Web-app # Streamlit-based UI for real-time fire/smoke detection
├── LICENSE # Project license

## 📦 Installation
git clone https://github.com/Hamza-1711/d-fire-detection.git
cd d-fire-detection

# Optional: Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r Web-app/requirements.txt

# ▶️ Run the App
cd Web-app
streamlit run app.py
Make sure you have:
RNN_model.h5 in Models
CNN_model.pt in Models
# 👤 Author
Developed with ❤️ by Muhammad Hamza
