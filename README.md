# 🛡️ Spam Classifier Pro

Advanced Machine Learning web application to classify emails/SMS as Spam or Legitimate with 97.76% accuracy.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

### 🔍 **Single Message Classification**
- Real-time spam detection
- Confidence score with visual gauge
- Animated results

### 📁 **Bulk Processing**
- Upload CSV/TXT files
- Paste multiple messages
- Download results

### 📊 **Analytics Dashboard**
- Live statistics
- Classification history
- Interactive charts

### 🎨 **Advanced UI**
- Dark/Light mode toggle
- Responsive design
- Share results (WhatsApp, Email)
- Download reports

## 🚀 Live Demo

**[Try it now!](https://spamclassifier2029.streamlit.app/)** ← (Update after deployment)

## 📸 Screenshots

*Screenshots coming soon*

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Clone Repository
```bash
git clone https://github.com/NamanSaxena2029/spam-classifier.git
cd spam-classifier
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📦 Dataset

**SMS Spam Collection** from Kaggle
- Total Messages: 5,572
- Ham (Legitimate): 4,825 (86.6%)
- Spam: 747 (13.4%)

📥 [Download Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## 🤖 Model Details

### Algorithm
- **Multinomial Naive Bayes**
- **TF-IDF Vectorization** (3000 features, bigrams)

### Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 97.76% |
| **Precision (HAM)** | 98% |
| **Precision (SPAM)** | 97% |
| **Recall (HAM)** | 100% |
| **Recall (SPAM)** | 86% |
| **F1-Score** | 98% |

### Training
```bash
python retrain_model.py
```

This will:
- Load original + custom datasets
- Train the model
- Save `model.pkl` and `vectorizer.pkl`

## 📁 Project Structure
```
spam-classifier/
│
├── app.py                      # Main Streamlit application
├── spam_classifier.ipynb       # Jupyter notebook (training & analysis)
├── retrain_model.py            # Model retraining script
│
├── model.pkl                   # Trained Naive Bayes model
├── vectorizer.pkl              # TF-IDF vectorizer
│
├── spam.csv                    # Original dataset (5,572 messages)
├── custom_spam_data.csv        # Custom training examples
│
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

## 🎨 Features Breakdown

### 1. Single Check Tab
- Enter message manually
- Quick example buttons
- Real-time classification
- Visual confidence gauge
- Share results (WhatsApp/Email/Copy)
- Download report

### 2. Bulk Check Tab
- Upload CSV/TXT files
- Paste multiple messages
- Batch processing with progress bar
- Download results as CSV

### 3. Analytics Tab
- Total classifications count
- Spam vs Ham distribution (pie chart)
- Full history table
- Clear history option

### 4. Sidebar Dashboard
- Live statistics
- Recent history (last 5)
- Distribution chart
- Model information

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository: `spam-classifier`
6. Main file: `app.py`
7. Click "Deploy"

**Live in 2 minutes!** 🚀

### Other Options
- **Heroku**: Add Procfile for deployment
- **Render**: Direct GitHub integration
- **AWS/GCP**: Docker deployment

## 🔧 Configuration

### Custom Training Data
Add your examples to `custom_spam_data.csv`:
```csv
label,message
ham,Your legitimate message here
spam,Your spam example here
```

Then retrain:
```bash
python retrain_model.py
```

## 📊 Technologies Used

- **Python 3.11**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **Plotly** - Interactive charts
- **NumPy** - Numerical computing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Naman Saxena**
- GitHub: [@NamanSaxena2029](https://github.com/NamanSaxena2029)
- LinkedIn: [Naman Saxena](https://linkedin.com/in/naman-saxena-61122126b/)

## 🙏 Acknowledgments

- Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Inspired by modern ML applications
- Built with ❤️ using Streamlit

## 📞 Support

For support:
- 📧 Email: namansaxena2029@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/NamanSaxena2029/spam-classifier/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/NamanSaxena2029/spam-classifier/discussions)

---

**⭐ If you found this project helpful, please give it a star!**

**🔗 Share with others who might find it useful!**
```