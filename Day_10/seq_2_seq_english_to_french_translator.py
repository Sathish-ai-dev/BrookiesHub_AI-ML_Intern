

import os
import io
import zipfile
import re
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

DATA_URL = "https://www.manythings.org/anki/fra-eng.zip" 
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_ZIP = DATA_DIR / "fra-eng.zip"
RAW_TXT = DATA_DIR / "fra.txt"

NUM_SAMPLES = 20000          
MAX_VOCAB_SIZE_IN = 10000    
MAX_VOCAB_SIZE_OUT = 10000  
EMBEDDING_DIM = 256
LSTM_UNITS = 256
BATCH_SIZE = 64
EPOCHS = 25                  

SAVED_DIR = Path("saved")
SAVED_DIR.mkdir(exist_ok=True)



def download_and_extract_dataset(url=DATA_URL, zip_path=RAW_ZIP, txt_path=RAW_TXT):
    """Download the fra-eng dataset and extract fra.txt"""
    if txt_path.exists():
        print(f"Using existing {txt_path}")
        return txt_path
    import requests
    print("Downloading dataset...")
    r = requests.get(url)
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        f.write(r.content)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        
        for name in z.namelist():
            if name.endswith('fra.txt') or name.endswith('fra-eng/fra.txt'):
                z.extract(name, DATA_DIR)
                extracted = DATA_DIR / name
                extracted.rename(txt_path)
                break
        else:
            raise FileNotFoundError('fra.txt not found inside zip')
    print(f"Saved to {txt_path}")
    return txt_path


def load_sentence_pairs(txt_path=RAW_TXT, num_samples=NUM_SAMPLES):
    """Read the .txt file and return list of (eng, fra) pairs.
    File format: English \t French \t (extra)
    We'll keep only the first 2 columns.
    """
    print("Loading sentence pairs...")
    pairs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            eng = parts[0].strip()
            fra = parts[1].strip()
            if eng == '' or fra == '':
                continue
            pairs.append((eng, fra))
    print(f"Loaded {len(pairs)} sentence pairs")
    return pairs


def simple_clean_text(text):
    
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



def prepare_tokenizers(pairs, num_words_in=MAX_VOCAB_SIZE_IN, num_words_out=MAX_VOCAB_SIZE_OUT):
    """Return fitted tokenizers for English (source) and French (target).
    For targets we add explicit start / end tokens: 'startseq' and 'endseq'
    """
    input_texts = []
    target_texts = []
    for eng, fra in pairs:
        eng_c = simple_clean_text(eng)
        fra_c = simple_clean_text(fra)
        
        fra_c = 'startseq ' + fra_c + ' endseq'
        input_texts.append(eng_c)
        target_texts.append(fra_c)

    
    tok_in = Tokenizer(num_words=num_words_in, filters='')  
    tok_in.fit_on_texts(input_texts)

    
    tok_out = Tokenizer(num_words=num_words_out, filters='')
    tok_out.fit_on_texts(target_texts)

    return tok_in, tok_out, input_texts, target_texts


def sequences_and_padding(tok_in, tok_out, input_texts, target_texts):
    input_seq = tok_in.texts_to_sequences(input_texts)
    target_seq = tok_out.texts_to_sequences(target_texts)

    max_enc_len = max(len(s) for s in input_seq)
    max_dec_len = max(len(s) for s in target_seq)
    print(f"Max encoder length: {max_enc_len}, Max decoder length: {max_dec_len}")

    encoder_input_data = pad_sequences(input_seq, maxlen=max_enc_len, padding='post')
    decoder_input_data = pad_sequences([s[:-1] for s in target_seq], maxlen=max_dec_len-1, padding='post')
    decoder_target_data = pad_sequences([s[1:] for s in target_seq], maxlen=max_dec_len-1, padding='post')

    return encoder_input_data, decoder_input_data, decoder_target_data, max_enc_len, max_dec_len-1



def build_training_model(inp_vocab_size, out_vocab_size, max_enc_len, max_dec_len,
                         embedding_dim=EMBEDDING_DIM, units=LSTM_UNITS):
    # Encoder
    encoder_inputs = Input(shape=(max_enc_len,), name='encoder_inputs')
    enc_emb = Embedding(inp_vocab_size, embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Decoder
    decoder_inputs = Input(shape=(max_dec_len,), name='decoder_inputs')
    dec_emb_layer = Embedding(out_vocab_size, embedding_dim, mask_zero=True, name='decoder_embedding')
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Luong-style attention (vectorized)
    # score = decoder_outputs @ encoder_outputs^T  -> shape (batch, dec_len, enc_len)
    score = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name='score')([decoder_outputs, encoder_outputs])
    attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name='attention_weights')(score)
    # context = attention_weights @ encoder_outputs -> (batch, dec_len, units)
    context = Lambda(lambda x: tf.matmul(x[0], x[1]), name='context')([attention_weights, encoder_outputs])

    # Concatenate context and decoder_outputs for each time step
    concat = Concatenate(axis=-1, name='concat')([context, decoder_outputs])  # (batch, dec_len, 2*units)

    # Final projection to vocabulary
    dense_time = TimeDistributed(Dense(out_vocab_size, activation='softmax'), name='time_dist_dense')
    outputs = dense_time(concat)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    return model, encoder_lstm, decoder_lstm, dec_emb_layer, dense_time

# Masked loss to ignore padding (0)
def masked_loss(y_true, y_pred):
    # y_true shape: (batch, dec_len)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# ----------------------
# Inference models
# ----------------------

def build_inference_models(encoder_inputs_shape, max_enc_len, max_dec_len, units,
                           encoder_lstm_layer, decoder_lstm_layer, decoder_embedding_layer, dense_time_layer):
    # encoder_inputs_shape is vocab-based (we need the encoder input placeholder for the encoder model)
    encoder_inputs = Input(shape=(max_enc_len,), name='enc_inputs_inf')
    enc_emb = encoder_lstm_layer.input._keras_history.layer.embeddings if False else None
    
    raise RuntimeError("This function is a placeholder — inference models are constructed below using returned layers.")




def run_full_pipeline(num_samples=NUM_SAMPLES, epochs=EPOCHS):
    txt = download_and_extract_dataset()
    pairs = load_sentence_pairs(txt, num_samples=num_samples)

    tok_in, tok_out, input_texts, target_texts = prepare_tokenizers(pairs)
    encoder_input_data, decoder_input_data, decoder_target_data, max_enc_len, max_dec_len = sequences_and_padding(tok_in, tok_out, input_texts, target_texts)

    inp_vocab_size = min(MAX_VOCAB_SIZE_IN, len(tok_in.word_index) + 1)
    out_vocab_size = min(MAX_VOCAB_SIZE_OUT, len(tok_out.word_index) + 1)
    print(f"Input vocab size: {inp_vocab_size}, Output vocab size: {out_vocab_size}")

    # Build model
    model, encoder_lstm_layer, decoder_lstm_layer, decoder_embedding_layer, dense_time_layer = build_training_model(
        inp_vocab_size, out_vocab_size, max_enc_len, max_dec_len, EMBEDDING_DIM, LSTM_UNITS)

    model.compile(optimizer=Adam(1e-3), loss=masked_loss, metrics=["accuracy"])
    model.summary()

    # Train
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=BATCH_SIZE, epochs=epochs, validation_split=0.1)

    # Save model and tokenizers
    model.save(SAVED_DIR / 'seq2seq_attention_model')
    import pickle
    with open(SAVED_DIR / 'tokenizer_in.pkl', 'wb') as f:
        pickle.dump(tok_in, f)
    with open(SAVED_DIR / 'tokenizer_out.pkl', 'wb') as f:
        pickle.dump(tok_out, f)
    print(f"Saved model and tokenizers to {SAVED_DIR}")

    # Build inference models (encoder_model and decoder_model) using layers from the trained graph
    # We will reconstruct inference models reusing the layers by name from the training model.

    # ----- Encoder inference model -----
    encoder_inputs = model.get_layer('encoder_inputs').input
    encoder_embedding = model.get_layer('encoder_embedding')
    encoder_lstm = model.get_layer('encoder_lstm')
    enc_emb = encoder_embedding(encoder_inputs)
    enc_outputs, enc_state_h, enc_state_c = encoder_lstm(enc_emb)
    encoder_model = Model(encoder_inputs, [enc_outputs, enc_state_h, enc_state_c])

    # ----- Decoder inference model (step-by-step) -----
    # Inputs for decoder one step
    decoder_single_input = Input(shape=(1,), name='decoder_single_input')
    decoder_state_input_h = Input(shape=(LSTM_UNITS,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(LSTM_UNITS,), name='decoder_state_input_c')
    encoder_outputs_input = Input(shape=(max_enc_len, LSTM_UNITS), name='encoder_outputs_input')

    # reuse embedding layer and decoder LSTM layer weights
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_lstm_layer = model.get_layer('decoder_lstm')
    time_dist_dense = model.get_layer('time_dist_dense')

    dec_emb_one = decoder_embedding_layer(decoder_single_input)  # (batch, 1, emb)
    dec_outputs_one, dec_state_h_one, dec_state_c_one = decoder_lstm_layer(dec_emb_one, initial_state=[decoder_state_input_h, decoder_state_input_c])

    # Attention: score = dec_output @ encoder_outputs^T -> (batch, 1, enc_len)
    score_one = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([dec_outputs_one, encoder_outputs_input])
    attn_weights_one = Lambda(lambda x: tf.nn.softmax(x, axis=-1))(score_one)
    context_one = Lambda(lambda x: tf.matmul(x[0], x[1]))([attn_weights_one, encoder_outputs_input])

    # concat context and dec_outputs_one
    # Both context_one and dec_outputs_one are (batch, 1, units); concatenate on last axis -> (batch, 1, 2*units)
    concat_one = Concatenate(axis=-1)([context_one, dec_outputs_one])

    # run through the same dense (TimeDistributed) layer. Since time_dist_dense expects a time-dimension,
    # we can apply its inner Dense by calling .layer if needed. Simpler: take the Dense layer inside time_dist_dense
    # TimeDistributed(Dense(...)) layer internals not directly exposed; but the TimeDistributed wrapper has a layer attribute.
    dense_layer = time_dist_dense.layer
    # Apply dense_layer to the concatenated vector at timestep
    # concat_one shape: (batch, 1, 2*units) -> we want output (batch, vocab)
    logits_one = dense_layer(tf.squeeze(concat_one, axis=1))  # shape (batch, vocab)
    output_probs = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name='output_probs')(logits_one)

    decoder_model = Model(
        [decoder_single_input, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
        [output_probs, dec_state_h_one, dec_state_c_one, attn_weights_one])

    # Save inference models for convenience
    encoder_model.save(SAVED_DIR / 'encoder_model')
    decoder_model.save(SAVED_DIR / 'decoder_model')

    print("Saved inference models.")

    return model, encoder_model, decoder_model, tok_in, tok_out, max_enc_len, max_dec_len

# ----------------------
# Helper for translating sentences with inference models
# ----------------------

def seq_to_text(seq, tokenizer):
    reverse_map = {v: k for k, v in tokenizer.word_index.items()}
    words = []
    for idx in seq:
        if idx == 0:
            continue
        w = reverse_map.get(idx, '')
        if w == 'startseq' or w == 'endseq':
            continue
        words.append(w)
    return ' '.join(words)


def translate_sentence(input_sentence, encoder_model, decoder_model, tokenizer_in, tokenizer_out, max_enc_len, max_dec_len):
    input_sentence = simple_clean_text(input_sentence)
    seq = tokenizer_in.texts_to_sequences([input_sentence])
    seq = pad_sequences(seq, maxlen=max_enc_len, padding='post')

    # Encode
    enc_outs, enc_h, enc_c = encoder_model.predict(seq)

    # Start token for decoder
    start_token = tokenizer_out.word_index.get('startseq')
    end_token = tokenizer_out.word_index.get('endseq')

    target_seq = np.array([[start_token]])
    stop_condition = False
    decoded_sentence = []
    state_h = enc_h
    state_c = enc_c

    for _ in range(max_dec_len):
        output_tokens, h, c, attn = decoder_model.predict([target_seq, state_h, state_c, enc_outs])
        # output_tokens shape (batch=1, vocab)
        sampled_token_index = np.argmax(output_tokens[0])
        if sampled_token_index == 0:
            break
        sampled_word = tokenizer_out.index_word.get(sampled_token_index, '')
        if sampled_word == 'endseq' or sampled_word == '':
            break
        decoded_sentence.append(sampled_word)

        # update target_seq
        target_seq = np.array([[sampled_token_index]])
        state_h, state_c = h, c

    return ' '.join(decoded_sentence)

# ----------------------
# If run as script: execute pipeline and show examples
# ----------------------

if __name__ == '__main__':
    # Run full pipeline (download, train, save). For quick tests, reduce NUM_SAMPLES and EPOCHS at top.
    model, encoder_model, decoder_model, tok_in, tok_out, max_enc_len, max_dec_len = run_full_pipeline()

    # Try a few translations
    tests = [
        "I am hungry.",
        "How are you?",
        "She is reading a book.",
        "Where is the bathroom?",
    ]
    for s in tests:
        print(f"EN: {s}")
        print("FR:", translate_sentence(s, encoder_model, decoder_model, tok_in, tok_out, max_enc_len, max_dec_len))
        print('---')

