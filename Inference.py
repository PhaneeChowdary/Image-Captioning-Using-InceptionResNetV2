import os
import json
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU, Add, Attention, Dense, Embedding,
    LayerNormalization, Reshape, StringLookup, TextVectorization
)

def create_encoder(img_height, img_width, img_channels, attention_dim):
    # Create feature extractor
    feature_extractor = InceptionResNetV2(include_top=False, weights="imagenet")
    feature_extractor.trainable = False
    
    # Create encoder
    image_input = Input(shape=(img_height, img_width, img_channels))
    image_features = feature_extractor(image_input)
    feature_shape = image_features.shape[1:]
    x = Reshape((feature_shape[0] * feature_shape[1], feature_shape[2]))(image_features)
    encoder_output = Dense(attention_dim, activation="relu")(x)
    return tf.keras.Model(inputs=image_input, outputs=encoder_output)

def create_decoder(vocab_size, max_caption_len, attention_dim):
    # Inputs
    word_input = Input(shape=(max_caption_len,), name="words")
    encoder_output = Input(shape=(None, attention_dim))
    gru_state_input = Input(shape=(attention_dim,), name="gru_state_input")
    
    # Layers
    embed_x = Embedding(vocab_size, attention_dim)(word_input)
    decoder_gru = GRU(
        attention_dim,
        return_sequences=True,
        return_state=True
    )
    decoder_attention = Attention()
    layer_norm = LayerNormalization(axis=-1)
    decoder_output_dense = Dense(vocab_size)
    
    # Forward pass
    gru_output, gru_state = decoder_gru(embed_x, initial_state=gru_state_input)
    context_vector = decoder_attention([gru_output, encoder_output])
    addition_output = Add()([gru_output, context_vector])
    layer_norm_output = layer_norm(addition_output)
    decoder_output = decoder_output_dense(layer_norm_output)
    
    return tf.keras.Model(
        inputs=[word_input, gru_state_input, encoder_output],
        outputs=[decoder_output, gru_state]
    )

def beam_search(image_features, decoder, tokenizer, config, beam_width=3):
    vocab = tokenizer.get_vocabulary()
    word_to_index = StringLookup(mask_token="", vocabulary=vocab)
    
    # Initialize
    gru_state = tf.zeros((1, config['ATTENTION_DIM']))
    start_token = word_to_index("<start>")
    
    initial_sequences = [([start_token], 0.0, gru_state)]
    completed_sequences = []

    for _ in range(config['MAX_CAPTION_LEN']):
        candidates = []

        for seq, score, curr_state in initial_sequences:
            if seq[-1] == word_to_index("<end>"):
                completed_sequences.append((seq, score))
                continue
                
            # Predict next tokens
            dec_input = tf.expand_dims([seq[-1]], 1)
            predictions, new_state = decoder(
                [dec_input, curr_state, image_features]
            )
            
            # Get top k predictions
            logits = predictions[0, 0]
            top_k_logits, top_k_indices = tf.math.top_k(logits, k=beam_width)
            
            # Add candidates
            for i in range(beam_width):
                candidate_seq = seq + [top_k_indices[i].numpy()]
                candidate_score = score - tf.math.log(tf.nn.softmax(logits)[top_k_indices[i]])
                candidates.append((candidate_seq, candidate_score.numpy(), new_state))
        
        # Select top sequences
        ordered = sorted(candidates, key=lambda x: x[1])
        initial_sequences = ordered[:beam_width]
    
    # Return best completed sequence or best incomplete sequence
    if completed_sequences:
        best_seq = sorted(completed_sequences, key=lambda x: x[1])[0][0]
    else:
        best_seq = initial_sequences[0][0]
    
    # Convert indices to words
    return [vocab[idx] for idx in best_seq]

def predict_caption(image_path, encoder, decoder, tokenizer, config):
    try:
        # Process image
        img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=config['IMG_CHANNELS'])
        img = tf.image.resize(img, (config['IMG_HEIGHT'], config['IMG_WIDTH']))
        img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
        features = encoder(tf.expand_dims(img, axis=0))
        
        # Generate caption using beam search
        caption_tokens = beam_search(features, decoder, tokenizer, config)
        
        # Remove start and end tokens and join
        caption = " ".join([word for word in caption_tokens[1:-1] if word not in ["<start>", "<end>"]])
        return caption + "."

    except Exception as e:
        print(f"Error in caption prediction: {str(e)}")
        return None

def display_prediction(image_path, caption):
    """Display the image and its generated caption."""
    try:
        plt.figure(figsize=(8, 8))
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        
        # Add caption with word wrapping
        plt.title(f"Generated Caption:\n{caption}", 
                 pad=20, 
                 wrap=True,
                 fontsize=12)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {str(e)}")

def predict_and_display_caption(image_path, encoder, decoder, tokenizer, config):
    try:
        # Process image and generate caption
        caption = predict_caption(image_path, encoder, decoder, tokenizer, config)
        
        if caption:
            # Display the image with the caption
            display_prediction(image_path, caption)
            return caption
        return None

    except Exception as e:
        print(f"Error in prediction and display: {str(e)}")
        return None

def load_models(model_dir='./Epochs10'):
    try:
        # Load configuration
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)

        # Load vocabulary
        with open(os.path.join(model_dir, 'tokenizer_vocab.json'), 'r') as f:
            vocab = json.load(f)
        tokenizer = TextVectorization(
            max_tokens=config['VOCAB_SIZE'],
            output_sequence_length=config['MAX_CAPTION_LEN']
        )
        tokenizer.set_vocabulary(vocab)

        # Create and load models
        encoder = create_encoder(
            config['IMG_HEIGHT'], 
            config['IMG_WIDTH'], 
            config['IMG_CHANNELS'],
            config['ATTENTION_DIM']
        )
        decoder = create_decoder(
            config['VOCAB_SIZE'],
            config['MAX_CAPTION_LEN'],
            config['ATTENTION_DIM']
        )

        # Load weights
        encoder.load_weights(os.path.join(model_dir, 'encoder.weights.h5'))
        decoder.load_weights(os.path.join(model_dir, 'decoder.weights.h5'))

        print("Models loaded successfully!")
        return encoder, decoder, tokenizer, config
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        print("Loading models...")
        encoder, decoder, tokenizer, config = load_models()
        
        while True:
            image_path = input("\nEnter image path (or 'q' to quit): ")
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                print(f"Error: Image file {image_path} not found")
                continue
             
            predict_and_display_caption(image_path, encoder, decoder, tokenizer, config)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")