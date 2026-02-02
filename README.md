  # 🏥 SmartMed AI – AI-Powered Medical Diagnosis Platform

  ![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-F7931E?logo=scikit-learn)
  ![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?logo=flask)
  ![Pandas](https://img.shields.io/badge/Pandas-2.1.4-150458?logo=pandas)
  ![NumPy](https://img.shields.io/badge/NumPy-1.24.4-013243?logo=numpy)
  ![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?logo=bootstrap)
  ![License](https://img.shields.io/badge/License-MIT-blue.svg)
  ![Status](https://img.shields.io/badge/Status-Active-brightgreen)
  ![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)

  > **⚠️ IMPORTANT**: SmartMed AI is an **educational/research prototype** designed to demonstrate AI capabilities in healthcare. It is **NOT** a certified medical device and **SHOULD NOT** be used for actual medical diagnosis or treatment decisions.

  ## 📌 Table of Contents
  - [Overview](#-overview)
  - [Screenshots](#-screenshots)
  - [System Architecture](#-system-architecture)
  - [Live Demo](#-live-demo)
  - [Quick Start](#-quick-start)
  - [Project Structure](#-project-structure)
  - [Model Performance](#-model-performance)
  - [API Documentation](#-api-documentation)
  - [Usage Examples](#-usage-examples)
  - [Technical Specifications](#-technical-specifications)
  - [Development Setup](#-development-setup)
  - [Contributing](#-contributing)
  - [Security & Privacy](#-security--privacy)
  - [Medical Disclaimer](#-medical-disclaimer)
  - [License](#-license)
  - [Contact & Support](#-contact--support)
  - [Acknowledgments](#-acknowledgments)

  ## 🎯 Overview

  SmartMed AI is an advanced machine learning-powered medical diagnosis system that predicts diseases from symptoms and provides comprehensive treatment recommendations. Built with state-of-the-art ML algorithms and a modern web interface, it serves as an educational platform demonstrating AI's potential in healthcare triage and decision support.

  ### Core Capabilities
  - **AI-Driven Diagnosis**: Predicts 41 diseases from 132 symptoms with high accuracy
  - **Comprehensive Care Plans**: Provides medications, precautions, diet, and exercise recommendations
  - **Modern Web Interface**: Responsive, user-friendly design with real-time feedback
  - **Local Deployment**: Runs entirely on your machine for privacy and security
  - **Educational Focus**: Designed for AI/ML learning and healthcare technology demonstrations

  ## 📸 Screenshots

![SmartMed1](screenshots/Smartmed1.png)
![SmartMed2](screenshots/Smartmed2.png)
![SmartMed3](screenshots/Smartmed3.png)
![SmartMed4](screenshots/Smartmed4.png)
![SmartMed5](screenshots/Smartmed5.png)
![SmartMed6](screenshots/Smartmed6.png)
![SmartMed7](screenshots/Smartmed7.png)


  ## 🏗️ System Architecture

  ```
  ┌───────────────────────┐
  │     Web Interface     │
  │   (Flask + Jinja2)    │
  │   ← renders index.html│
  └───────────┬───────────┘
              │
  ┌───────────▼───────────┐
  │   Prediction Engine   │
  │  • get_predicted_value()│
  │  • helper()           │
  └───────────┬───────────┘
              │
  ┌───────────▼───────────┐
  │   Data Layer          │
  │  • svc.pkl (SVC model)│
  │  • CSV knowledge base │
  └───────────────────────┘
  ```

  ### Architecture Components

  | Layer | Component | Description |
  |-------|-----------|-------------|
  | **Presentation Layer** | Web Interface | User-facing UI built with Flask and Bootstrap, renders HTML templates |
  | **Application Layer** | Prediction Engine | Core logic for processing symptoms and generating predictions |
  | **Data Layer** | ML Model & Knowledge Base | Pre-trained SVC model and CSV files containing medical knowledge |

  ### Data Flow
  1. **User Input** → Symptoms entered via web interface
  2. **Processing** → Prediction Engine processes input using helper functions
  3. **Model Inference** → SVC model makes disease prediction
  4. **Knowledge Retrieval** → Relevant treatment data fetched from CSV files
  5. **Result Rendering** → Comprehensive diagnosis displayed in web interface

  ## 🌐 Live Demo

  **Local Deployment Only**: This application runs on your local machine for maximum privacy and control.

  ### Access Points:
  - **Main Application**: [http://localhost:5000](http://localhost:5000) (after starting the server)
  - **Default Port**: 5000 (configurable)

  ## 🚀 Quick Start

  ### Prerequisites
  - **Python 3.8+**
  - **pip** (Python package manager)
  - **Git** (for cloning repository)

  ### Installation Steps

  #### 1. Clone the Repository
  ```bash
  git clone https://github.com/boughanmiyoussef/SmartMed.git
  cd SmartMed
  ```

  #### 2. Create Virtual Environment (Recommended)
  ```bash
  # Windows
  python -m venv venv
  venv\Scripts\activate

  # Linux/Mac
  python3 -m venv venv
  source venv/bin/activate
  ```

  #### 3. Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

  #### 4. Launch Application
  ```bash
  python main.py
  ```

  #### 5. Access the Application
  Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

  ## 📁 Project Structure

  ```
  SmartMed/
  ├── main.py                      # Flask application entry point
  ├── requirements.txt             # Python dependencies
  ├── README.md                    # This documentation
  ├── .gitignore                   # Git ignore rules
  │
  ├── models/                      # Machine Learning models
  │   ├── svc.pkl                  # Trained Support Vector Classifier
  │   └── SmartMed.ipynb           # Jupyter notebook (training & exploration)
  │
  ├── datasets/                    # Medical knowledge base
  │   ├── Training.csv            # 4,920 synthetic patient records
  │   ├── description.csv         # Disease descriptions
  │   ├── precautions_df.csv      # Medical precautions
  │   ├── medications.csv         # Recommended medications
  │   ├── diets.csv               # Diet recommendations
  │   └── workout_df.csv          # Exercise plans
  │
  └── templates/                   # Web interface template
      └── index.html              # Main diagnosis page
  ```

  ### Dataset Composition
  | Metric | Value | Description |
  |--------|-------|-------------|
  | **Total Records** | 4,920 | Synthetic patient cases |
  | **Number of Diseases** | 41 | Medical conditions covered |
  | **Number of Symptoms** | 132 | Binary symptom features |
  | **Average Symptoms per Case** | 8.4 | Mean symptom count |
  | **Class Distribution** | Uniform | 120 cases per disease |
  | **Missing Values** | 0% | Complete dataset |


  ### Data Characteristics
  - **Synthetic Data**: Generated for educational purposes
  - **Binary Features**: Symptoms represented as 0/1 (absent/present)
  - **Clean Dataset**: No missing values or inconsistencies
  - **Balanced Classes**: Equal representation of all diseases

  ## 📈 Model Performance

  ### Training Results
  | Model | Accuracy | Precision | Recall | F1-Score | Training Time |
  |-------|----------|-----------|--------|----------|---------------|
  | **Support Vector Classifier (SVC)** | **100%** | **1.0000** | **1.0000** | **1.0000** | 2.3s |

  **Note**: The 100% accuracy reflects perfect separation in this synthetic, clean dataset for educational purposes.

  ### Performance Metrics
  | Metric | Value | Notes |
  |--------|-------|-------|
  | **Response Time** | < 50ms average | From symptom input to diagnosis |
  | **Memory Usage** | < 130 MB | Peak memory consumption |
  | **Model Size** | 2.3 MB | Compressed pickle file |
  | **Startup Time** | 2-3 seconds | Model loading and initialization |

  ## 📡 API Documentation

  ### Base URL
  ```
  http://localhost:5000
  ```

  ### Endpoints

  #### 1. Home Page (GET /)
  ```http
  GET http://localhost:5000/
  ```
  - Renders the main diagnosis interface
  - Displays input form for symptoms

  #### 2. Disease Prediction (POST /predict)
  ```http
  POST http://localhost:5000/predict
  Content-Type: application/x-www-form-urlencoded
  ```

  **Parameters:**
  - `symptoms`: Comma-separated list of symptoms (e.g., "headache, fever, cough")

  **Response:**
  - Returns HTML page with diagnosis and treatment recommendations

  **Example using curl:**
  ```bash
  curl -X POST http://localhost:5000/predict \
    -d "symptoms=headache,fever,cough"
  ```

  ## 🎯 Usage Examples

  ### Web Interface Usage
  1. **Access the Application**: Navigate to `http://localhost:5000`
  2. **Enter Symptoms**: Type symptoms in the search box (e.g., "headache fever cough")
  3. **Submit for Analysis**: Click "Start AI Diagnosis"
  4. **Review Results**: View predicted disease and complete treatment plan
  5. **Start New Diagnosis**: Click "Start New Diagnosis" to clear and begin again

  ## ⚙️ Technical Specifications

  ### System Requirements
  | Component | Minimum | Recommended |
  |-----------|---------|-------------|
  | Python | 3.8+ | 3.10+ |
  | RAM | 512 MB | 2 GB |
  | Storage | 50 MB | 100 MB |
  | CPU | 1 Core | 2+ Cores |

  ### Dependencies
  ```txt
  Flask==2.3.3
  scikit-learn==1.3.2
  pandas==2.1.4
  numpy==1.24.4
  joblib==1.3.2
  ```

  ## 🔧 Development Setup

  ### For Developers

  #### 1. Clone with SSH
  ```bash
  git clone git@github.com:boughanmiyoussef/SmartMed.git
  cd SmartMed
  ```

  #### 2. Setup Development Environment
  ```bash
  # Create virtual environment
  python -m venv .venv

  # Activate (Windows)
  .venv\Scripts\activate

  # Activate (Linux/Mac)
  source .venv/bin/activate

  # Install development dependencies
  pip install -r requirements.txt
  pip install black flake8 pytest
  ```

  ## 🤝 Contributing

  We welcome contributions! Here's how you can help:

  ### Ways to Contribute
  1. **Report Bugs**: Open an issue with detailed information
  2. **Suggest Features**: Propose new features or improvements
  3. **Code Contributions**: Submit pull requests
  4. **Documentation**: Improve docs or add translations
  5. **Testing**: Help test and report issues

  ### Contribution Process
  1. Fork the repository
  2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
  3. Commit changes (`git commit -m 'Add AmazingFeature'`)
  4. Push to branch (`git push origin feature/AmazingFeature`)
  5. Open a Pull Request

  ## 🔒 Security & Privacy

  ### Data Protection
  - **Local Processing**: All data processed locally, no external transmission
  - **No Personal Data**: System doesn't collect or store personal information
  - **Synthetic Dataset**: Only uses publicly available synthetic data
  - **Stateless Design**: No user tracking or session storage

  ## ⚠️ Medical Disclaimer

  **CRITICAL INFORMATION – PLEASE READ CAREFULLY**

  ### Important Notice
  SmartMed AI is an **educational and research prototype** designed for:
  - Demonstrating AI/ML capabilities in healthcare
  - Academic research and learning
  - Technology demonstration purposes

  ### What This System IS NOT
  - **NOT** a certified medical device
  - **NOT** a replacement for professional medical advice
  - **NOT** validated for clinical use
  - **NOT** approved for medical diagnosis
  - **NOT** a substitute for qualified healthcare providers

  **ALWAYS CONSULT QUALIFIED HEALTHCARE PROFESSIONALS FOR MEDICAL ADVICE, DIAGNOSIS, AND TREATMENT.**

  ## 📄 License

  This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

  ```text
  MIT License

  Copyright (c) 2024 Youssef Boughanmi

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

  ## 📞 Contact & Support

  ### Primary Contact
  - **Name**: Youssef Boughanmi
  - **Email**: yussefboughanmy@gmail.com
  - **GitHub**: [@boughanmiyoussef](https://github.com/boughanmiyoussef)

  ### Project Links
  - **Repository**: https://github.com/boughanmiyoussef/SmartMed
  - **Issues**: https://github.com/boughanmiyoussef/SmartMed/issues
  - **Discussions**: https://github.com/boughanmiyoussef/SmartMed/discussions

  ## 🙏 Acknowledgments

  ### Organizations & Communities
  - **Holberton School Tunis** – For fostering applied AI education
  - **Open Source Community** – For invaluable tools and libraries
  - **Medical Research Community** – For inspiration and guidance

  ### Technologies & Tools
  - **Flask & Python Ecosystem** – Web framework and core technology
  - **scikit-learn** – Machine learning algorithms
  - **Pandas & NumPy** – Data processing and analysis
  - **Bootstrap & Font Awesome** – UI components and icons
  - **Jupyter Notebook** – Research and development environment

  ---

  <div align="center">

  ## 🎓 Educational Purpose Statement

  **SmartMed AI exists to advance understanding of AI in healthcare through responsible, transparent innovation.**

  *This project demonstrates classical machine learning applications in healthcare while emphasizing ethical considerations and professional medical consultation.*

  ---

  **Made with ❤️ for the advancement of medical AI education**

  *Local AI-powered diagnosis demonstration*

  </div>

  ---