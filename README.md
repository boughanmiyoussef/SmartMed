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
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Dataset Details](#-dataset-details)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [Technical Specifications](#-technical-specifications)
- [Development Setup](#-development-setup)
- [Testing](#-testing)
- [Future Roadmap](#-future-roadmap)
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

## ✨ Key Features

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0;">
  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
    <h4>🤖 Classical ML Algorithms</h4>
    <p>Uses Support Vector Classifier (SVC) for disease prediction - a proven algorithm for classification tasks. This project focuses on classical ML rather than deep learning for interpretability.</p>
  </div>
  
  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
    <h4>💻 Web Interface</h4>
    <p>Clean, intuitive interface with real-time symptom input and results display built with Flask and Bootstrap.</p>
  </div>
  
  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
    <h4>📊 Comprehensive Dataset</h4>
    <p>4,920 synthetic patient records covering 41 diseases and 132 symptoms. Binary feature representation for efficient processing.</p>
  </div>
  
  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
    <h4>🔧 Technical Excellence</h4>
    <p>Modular architecture with clean separation of concerns, fast response times, and low memory footprint.</p>
  </div>
</div>

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
├── templates/                   # Web interface templates
│   ├── index.html              # Main diagnosis page
│   ├── about.html              # About page
│   ├── blog.html               # Blog/Articles
│   ├── contact.html            # Contact form
│   └── developer.html          # Developer information
│
└── static/                      # Static assets (CSS, JS, images)
    ├── css/
    ├── js/
    └── images/
```

## 🏗️ System Architecture

<div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #f9f9f9; margin: 20px 0;">
  <div style="text-align: center; font-weight: bold; margin-bottom: 10px; color: #2c3e50;">Presentation Layer</div>
  <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
    <div style="padding: 15px; background: white; border-radius: 5px; border: 1px solid #3498db; min-width: 200px; text-align: center;">
      <strong>Web UI</strong><br>
      <small>Flask + Jinja2 + Bootstrap</small>
    </div>
  </div>

  <div style="text-align: center; margin: 20px 0;">
    <div style="display: inline-block; padding: 5px 15px; background: #3498db; color: white; border-radius: 20px;">▼</div>
  </div>

  <div style="text-align: center; font-weight: bold; margin-bottom: 10px; color: #2c3e50;">Business Logic Layer</div>
  <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
    <div style="padding: 15px; background: white; border-radius: 5px; border: 1px solid #e74c3c; min-width: 200px; text-align: center;">
      <strong>Prediction Engine</strong><br>
      <small>SVC Model Inference</small>
    </div>
    <div style="padding: 15px; background: white; border-radius: 5px; border: 1px solid #e74c3c; min-width: 200px; text-align: center;">
      <strong>Recommendation Engine</strong><br>
      <small>Treatment Plan Generator</small>
    </div>
  </div>

  <div style="text-align: center; margin: 20px 0;">
    <div style="display: inline-block; padding: 5px 15px; background: #3498db; color: white; border-radius: 20px;">▼</div>
  </div>

  <div style="text-align: center; font-weight: bold; margin-bottom: 10px; color: #2c3e50;">Data Access Layer</div>
  <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <div style="padding: 15px; background: white; border-radius: 5px; border: 1px solid #27ae60; min-width: 200px; text-align: center;">
      <strong>ML Models</strong><br>
      <small>Pickle Serialization</small>
    </div>
    <div style="padding: 15px; background: white; border-radius: 5px; border: 1px solid #27ae60; min-width: 200px; text-align: center;">
      <strong>Medical Database</strong><br>
      <small>CSV Knowledge Base</small>
    </div>
  </div>
</div>

## 🛠️ Technology Stack

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
  <div style="border: 1px solid #3498db; border-radius: 8px; padding: 15px; text-align: center; background: white;">
    <strong>Flask 2.3.3</strong><br>
    <small>Lightweight web framework</small>
  </div>
  <div style="border: 1px solid #e74c3c; border-radius: 8px; padding: 15px; text-align: center; background: white;">
    <strong>Scikit-learn 1.3.2</strong><br>
    <small>SVC ML algorithms</small>
  </div>
  <div style="border: 1px solid #f39c12; border-radius: 8px; padding: 15px; text-align: center; background: white;">
    <strong>Pandas 2.1.4</strong><br>
    <small>Data manipulation</small>
  </div>
  <div style="border: 1px solid #9b59b6; border-radius: 8px; padding: 15px; text-align: center; background: white;">
    <strong>NumPy 1.24.4</strong><br>
    <small>Numerical computing</small>
  </div>
  <div style="border: 1px solid #1abc9c; border-radius: 8px; padding: 15px; text-align: center; background: white;">
    <strong>Bootstrap 5.3</strong><br>
    <small>Frontend framework</small>
  </div>
  <div style="border: 1px solid #34495e; border-radius: 8px; padding: 15px; text-align: center; background: white;">
    <strong>Joblib 1.3.2</strong><br>
    <small>Model serialization</small>
  </div>
</div>

### ML Approach
- **Algorithm**: Support Vector Classifier (SVC) with linear kernel
- **Type**: Supervised learning for multi-class classification
- **Architecture**: Classical machine learning (not deep learning)
- **Focus**: Interpretability and efficiency over complex neural networks

## 📊 Dataset Details

### Dataset Composition
| Metric | Value | Description |
|--------|-------|-------------|
| **Total Records** | 4,920 | Synthetic patient cases |
| **Number of Diseases** | 41 | Medical conditions covered |
| **Number of Symptoms** | 132 | Binary symptom features |
| **Average Symptoms per Case** | 8.4 | Mean symptom count |
| **Class Distribution** | Uniform | 120 cases per disease |
| **Missing Values** | 0% | Complete dataset |

### Dataset Files
<div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 20px 0; background: #f8f9fa;">
  <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    <span style="padding: 8px 12px; background: #e3f2fd; border-radius: 5px; font-size: 0.9em;">Training.csv</span>
    <span style="padding: 8px 12px; background: #f3e5f5; border-radius: 5px; font-size: 0.9em;">description.csv</span>
    <span style="padding: 8px 12px; background: #e8f5e9; border-radius: 5px; font-size: 0.9em;">precautions_df.csv</span>
    <span style="padding: 8px 12px; background: #fff3e0; border-radius: 5px; font-size: 0.9em;">medications.csv</span>
    <span style="padding: 8px 12px; background: #fce4ec; border-radius: 5px; font-size: 0.9em;">diets.csv</span>
    <span style="padding: 8px 12px; background: #e0f2f1; border-radius: 5px; font-size: 0.9em;">workout_df.csv</span>
  </div>
</div>

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
| Random Forest | 100% | 1.0000 | 1.0000 | 1.0000 | 4.8s |
| Gradient Boosting | 100% | 1.0000 | 1.0000 | 1.0000 | 6.1s |
| K-Nearest Neighbors | 100% | 1.0000 | 1.0000 | 1.0000 | 0.8s |

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

#### 3. Debug Endpoints
```http
GET /test-symptom/<symptom>           # Test symptom formatting
GET /debug/<disease>                  # View parsed disease data
GET /doctor/<filename>                # Serve doctor images
```

## 🎯 Usage Examples

### Web Interface Usage
1. **Access the Application**: Navigate to `http://localhost:5000`
2. **Enter Symptoms**: Type symptoms in the search box (e.g., "headache fever cough")
3. **Submit for Analysis**: Click "Start AI Diagnosis"
4. **Review Results**: View predicted disease and complete treatment plan
5. **Start New Diagnosis**: Click "Start New Diagnosis" to clear and begin again

### Python Integration
```python
from main import get_predicted_value, helper

# Example 1: Basic disease prediction
symptoms = ['headache', 'fever', 'fatigue', 'cough']
disease = get_predicted_value(symptoms)
print(f"Predicted Disease: {disease}")

# Example 2: Complete treatment plan
description, precautions, medications, diets, workouts = helper(disease)

print("\n=== COMPLETE TREATMENT PLAN ===")
print(f"Disease: {disease}")
print(f"Description: {description}")
print(f"Precautions: {', '.join(precautions)}")
print(f"Medications: {', '.join(medications)}")
print(f"Diet Recommendations: {', '.join(diets)}")
print(f"Workout Suggestions: {', '.join(workouts)}")
```

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

#### 3. Code Style & Formatting
```bash
# Format code with Black
black .

# Check code style
flake8 .

# Run tests
pytest
```

### Project Configuration

#### Environment Variables
Create a `.env` file:
```env
FLASK_APP=main.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=True
PORT=5000
```

#### Running in Development Mode
```bash
# Enable debug mode
export FLASK_ENV=development
export FLASK_DEBUG=1

# Run with auto-reload
flask run --reload
```

## 🧪 Testing

### Test Structure
```
tests/
├── test_models.py      # ML model tests
├── test_api.py        # API endpoint tests
├── test_web.py        # Web interface tests
└── conftest.py        # Test configuration
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_api.py

# Run with coverage report
python -m pytest --cov=main --cov-report=html
```

## 🛣️ Future Roadmap

### Phase 1: Enhanced AI Capabilities
- [ ] **Explainable AI (XAI)**: SHAP/LIME for prediction explanations
- [ ] **Confidence Scores**: Probability estimates for predictions

### Phase 2: Platform Expansion
- [ ] **Multi-language Support**: Internationalization
- [ ] **API Rate Limiting**: Production-grade API management

### Phase 3: Advanced Features
- [ ] **Real-time Updates**: Live medical guideline integration
- [ ] **Patient History**: Secure symptom history tracking

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

### Security Features
- **Input Validation**: Sanitize all user inputs
- **Secure Headers**: Implement security best practices
- **Regular Updates**: Keep dependencies updated

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
- **LinkedIn**: [Youssef Boughanmi](https://www.linkedin.com/in/youssef-boughanmi-4990222a0)

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