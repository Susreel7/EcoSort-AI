# EcoSort AI (TrashFormer) ♻️

**EcoSort AI** is a smart waste classification system powered by Artificial Intelligence. It uses Deep Learning (MobileNetV2) to accurately classify different types of waste, helping users sort trash effectively for recycling and proper disposal.

The application features a modern web interface built with **Flask**, allowing users to upload images, use their camera for real-time detection, or process batches of images.

## 🚀 Features

*   **AI-Powered Classification**: Classifies waste into 7 categories:
    *   📦 Cardboard
    *   🔌 E-waste
    *   🍶 Glass
    *   🏥 Medical
    *   🥫 Metal
    *   📄 Paper
    *   🔄 Plastic
*   **Multiple Input Methods**:
    *   📸 **Camera Capture**: Real-time classification using your device's camera.
    *   📤 **Single Upload**: Drag and drop images for instant analysis.
    *   📚 **Batch Processing**: Upload multiple images at once for bulk classification.
    *   🎥 **Live Detection**: Real-time video stream analysis with Grad-CAM visualization (heatmaps).
*   **Detailed Analytics**: Track classification history and export reports (CSV/PDF).
*   **Educational Insights**: Provides recycling tips and disposal guidelines for each category.

## 🛠️ Tech Stack

*   **Backend**: Python 3.11, Flask
*   **Machine Learning**: TensorFlow, Keras (MobileNetV2 Transfer Learning)
*   **Image Processing**: Pillow (PIL), NumPy
*   **Frontend**: HTML5, CSS3, JavaScript

## 📂 Project Structure

```
EcoSort-AI/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── STARTUP.txt             # Quick startup guide
├── models/                 # Directory for trained models (.keras)
├── static/                 # CSS, JS, and images
├── templates/              # HTML templates
├── uploads/                # Temporary storage for uploaded images
└── scripts/                # Utility scripts for data processing & training
    ├── clean_dataset.py
    ├── split_data.py
    └── train_trashformer.py
```

## 📊 Dataset

This project uses the **TrashBox** dataset.
*   **Source**: [TrashBox Dataset on GitHub](https://github.com/nikhilvenkatkumsetty/TrashBox/tree/main/TrashBox_train_set)
*   **Categories**: Cardboard, E-waste, Glass, Medical, Metal, Paper, Plastic.

## ⚙️ Installation & Setup

### Prerequisites
*   Python 3.11 or higher
*   Git

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Susreel7/EcoSort-AI.git
    cd EcoSort-AI
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Start the Application**
    ```bash
    python app.py
    ```

2.  **Access the Web Interface**
    Open your web browser and navigate to:
    `http://localhost:5000`

## 🧠 Training Custom Models (Optional)

If you want to train the model yourself:

1.  **Prepare Data**:
    *   Download the dataset and place it in a `waste_data` folder one level up from the project directory (`../waste_data`).
    *   Ensure folders are named: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`, etc.

2.  **Split Data**:
    ```bash
    python scripts/split_data.py
    ```

3.  **Clean Dataset** (Optional but recommended):
    ```bash
    python scripts/clean_dataset.py
    ```

4.  **Train Model**:
    ```bash
    python scripts/train_trashformer.py
    ```
    The trained model will be saved in the `models/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
