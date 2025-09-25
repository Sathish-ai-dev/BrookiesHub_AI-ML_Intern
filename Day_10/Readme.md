# 🧠 English–French Neural Translation with LSTM + Attention

This project demonstrates how to build a neural machine translation system using an encoder–decoder architecture with attention mechanism. It translates English sentences into French using a small parallel corpus and deep learning techniques.

## 🎯 Objective

Build a sequence-to-sequence translation model using:
- Python
- Keras (TensorFlow backend)
- LSTM layers
- Attention mechanism

## 📦 Tools & Libraries

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Jupyter Notebook

## 📁 Project Structure

english-french-translation/ ├── translation_model.ipynb # Full training and demo notebook ├── data/ │ └── fra.txt # English–French sentence pairs ├── saved_model/ │ ├── encoder_model.h5 │ ├── decoder_model.h5 │ └── tokenizer.pkl └── README.md

Code

## 📚 Dataset

We use the [fra.txt](https://www.manythings.org/anki/fra-eng.zip) file from the Tatoeba Project, containing thousands of English–French sentence pairs. Each line is tab-separated:
Go. Va ! Hi. Salut !

## Code

## 🧪 Model Architecture

- **Encoder**: Embedding + LSTM
- **Decoder**: Embedding + LSTM
- **Attention Layer**: Computes context vectors to focus on relevant encoder outputs
- **Output Layer**: Dense softmax over French vocabulary

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/english-french-translation.git
   cd english-french-translation
   Install dependencies

   bash
    - pip install -r requirements.txt
    - Download and extract dataset
    - Download fra-eng.zip
Extract fra.txt into the data/ folder

Run the notebook Open translation_model.ipynb in Jupyter and run all cells to:

Preprocess data

Train the model

Save encoder/decoder/tokenizer

Test translation with demo inputs

# 🧠 Sample Translation
python
translate("How are you?")
# Output: "Comment ça va ?"
📈 Training Tips
Use 10k–30k sentence pairs for faster training

Tune embedding size, LSTM units, and batch size

Add dropout or regularization for better generalization

# 💡 Outcome
Learn how deep learning handles language translation

Understand encoder–decoder architecture and attention

Build a working demo of neural machine translation

# 📜 License
This project is open-source under the MIT License.

# 🙌 Acknowledgments
ManyThings.org for the dataset

TensorFlow/Keras for the deep learning framework