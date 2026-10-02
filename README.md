# 🧠 SentimentMind AI

**SentimentMind AI** is an AI-powered sentiment analysis application built using **RoBERTa** and **Streamlit**.

The application analyzes text and classifies its sentiment into three categories:

* 🔴 **Negative**
* 🟡 **Neutral**
* 🟢 **Positive**

It also provides confidence scores for each sentiment category.

---

## 🚀 Features

* 🤖 RoBERTa-based Transformer model
* 📝 Text sentiment analysis
* 🔴 Negative / 🟡 Neutral / 🟢 Positive classification
* 📊 Confidence scores
* ⚡ Streamlit interactive web interface
* 💻 CPU and GPU support
* 🎨 Clean and responsive UI

---

## 🧠 Model

SentimentMind AI uses:

* **Model:** RoBERTa-base
* **Task:** 3-class text classification
* **Classes:** Negative, Neutral, Positive
* **Maximum sequence length:** 128 tokens
* **Framework:** PyTorch
* **Transformers:** Hugging Face Transformers

### Test Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **78.73%** |
| F1 Score  | **78.52%** |
| Precision | **78.72%** |
| Recall    | **78.73%** |

The reported results are based on the held-out test dataset used during model evaluation.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* RoBERTa
* Streamlit
* NumPy

---

## 📁 Project Structure

```text
SentimentMind-AI/
│
├── app.py
├── style.css
├── requirements.txt
│
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/SentimentMind-AI.git
```

Move into the project directory:

```bash
cd SentimentMind-AI
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💡 Example

### Input

```text
I absolutely love this product! The quality is amazing.
```

### Output

```text
Predicted Sentiment: Positive

Negative: 2.15%
Neutral: 4.83%
Positive: 93.02%
```

*Example output only; actual confidence scores depend on the model prediction.*

---

## 🔮 Future Improvements

* Improve classification performance
* Add batch sentiment analysis
* Add CSV file upload
* Add sentiment visualization
* Deploy the application online
* Add multilingual sentiment analysis

---

## 👨‍💻 Author

**Muhammad Saim**

AI/ML Developer

---

## 📄 License

This project is intended for educational and portfolio purposes.
