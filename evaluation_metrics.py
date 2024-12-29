import nltk
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Import your existing functions
from Inference import (
    load_models,
    predict_and_display_caption,
    beam_search
)

# Download required NLTK data
try:
    nltk.download('punkt')
except:
    pass

def get_image_label(example):
    img = example["image"]
    img = tf.image.resize(img, (299, 299))
    img = img / 255
    return {"image_tensor": img, "captions": example["captions"]["text"]}

def calculate_bleu_scores(reference_captions, generated_caption):
    """Calculate BLEU scores for a single caption"""
    try:
        # Tokenize the generated caption
        hypothesis = word_tokenize(generated_caption.lower())

        # Tokenize all reference captions
        references = [word_tokenize(ref.lower()) for ref in reference_captions]
        
        # Use smoothing to avoid zero scores
        smoothing = SmoothingFunction().method1

        # Calculate BLEU scores
        bleu_1 = sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return bleu_1, bleu_2, bleu_3, bleu_4
    except Exception as e:
        print(f"Error calculating BLEU scores: {str(e)}")
        return 0, 0, 0, 0

def evaluate_on_test_dataset(encoder, decoder, tokenizer, config, num_samples=1000):
    """Evaluate model on test dataset"""
    try:
        # Load test dataset
        print("Loading test dataset...")
        testds = tfds.load("coco_captions", split="test", data_dir="gs://asl-public/data/tensorflow_datasets/")
        testds = testds.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        
        bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
        print(f"\nStarting evaluation on {num_samples} samples...")
        
        for i, data in enumerate(testds):
            if i >= num_samples:
                break
                
            try:
                # Get image and reference captions
                image_tensor = data["image_tensor"]
                reference_captions = [cap.numpy().decode('utf-8') for cap in data["captions"]]

                # Generate caption
                features = encoder(tf.expand_dims(image_tensor, 0))
                generated_caption_tokens = beam_search(features, decoder, tokenizer, config)
                generated_caption = " ".join([word for word in generated_caption_tokens[1:-1] 
                                           if word not in ["<start>", "<end>"]])
                
                # Calculate BLEU scores
                bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(reference_captions, generated_caption)
                
                # Store scores
                bleu_scores['bleu_1'].append(bleu_1)
                bleu_scores['bleu_2'].append(bleu_2)
                bleu_scores['bleu_3'].append(bleu_3)
                bleu_scores['bleu_4'].append(bleu_4)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} images")
                    
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
        
        # Calculate average scores and standard deviations
        avg_scores = {
            'bleu_1': np.mean(bleu_scores['bleu_1']),
            'bleu_2': np.mean(bleu_scores['bleu_2']),
            'bleu_3': np.mean(bleu_scores['bleu_3']),
            'bleu_4': np.mean(bleu_scores['bleu_4'])
        }
        
        std_scores = {
            'bleu_1_std': np.std(bleu_scores['bleu_1']),
            'bleu_2_std': np.std(bleu_scores['bleu_2']),
            'bleu_3_std': np.std(bleu_scores['bleu_3']),
            'bleu_4_std': np.std(bleu_scores['bleu_4'])
        }
        
        return avg_scores, std_scores, bleu_scores
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None, None, None

def plot_bleu_distributions(bleu_scores):
    """Plot distributions of BLEU scores"""
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4'], 1):
        plt.subplot(2, 2, i)
        plt.hist(bleu_scores[metric], bins=50)
        plt.title(f'Distribution of {metric.upper()} Scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('bleu_score_distributions.png')
    plt.show()

def save_results(avg_scores, std_scores, filename='evaluation_results.txt'):
    """Save evaluation results to file"""
    with open(filename, 'w') as f:
        f.write("Image Captioning Evaluation Results\n")
        f.write("==================================\n\n")
        f.write("Average BLEU Scores:\n")
        for metric, score in avg_scores.items():
            f.write(f"{metric.upper()}: {score:.4f} ± {std_scores[metric + '_std']:.4f}\n")

if __name__ == "__main__":
    try:
        # Load models
        print("Loading models...")
        encoder, decoder, tokenizer, config = load_models()
        
        # Run evaluation
        avg_scores, std_scores, bleu_scores = evaluate_on_test_dataset(
            encoder, decoder, tokenizer, config, num_samples=1000
        )
        
        if avg_scores and std_scores:
            # Print results
            print("\nEvaluation Results:")
            print("==================")
            for metric in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
                print(f"{metric.upper()}: {avg_scores[metric]:.4f} ± {std_scores[metric + '_std']:.4f}")
            
            # Save results
            save_results(avg_scores, std_scores)
            
            # Plot distributions
            plot_bleu_distributions(bleu_scores)
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error in main: {str(e)}")