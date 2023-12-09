import numpy as np

import matplotlib.pyplot as plt
import random

# Set a random seed for reproducibility
np.random.seed(42)

def read_lines_from_txt(file_path):
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.strip())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines


def p_stable_hash(x, a, b, p):
    result = int((a * x + b) % p)
    return result


def generate_p_stable_hash_functions(k, p):
    return [(np.random.randint(1, p), np.random.randint(0, p)) for _ in range(k)]


def minhash_signature(string, hash_functions, p, n=6):
    signature = np.inf * np.ones(len(hash_functions), dtype=int)
    for i, hash_func_params in enumerate(hash_functions):
        for j in range(len(string) - n + 1):
            token = string[j:j+n]

            hash_value = p_stable_hash(hash(token), *hash_func_params, p)
            signature[i] = min(signature[i], hash_value)
    return signature


def jaccard_similarity(signature1, signature2):
    intersection_size = np.sum(signature1 == signature2)
    union_size = len(signature1)
    return intersection_size / union_size


def average_pairwise_similarity(strings, k, p):
    hash_functions = generate_p_stable_hash_functions(k, p)
    signatures = [minhash_signature(string, hash_functions, p) for string in strings]
    
    total_similarity = 0
    total_pairs = 0

    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            similarity = jaccard_similarity(signatures[i], signatures[j])
            total_similarity += similarity
            total_pairs += 1

    average_similarity = total_similarity / total_pairs
    return average_similarity


def p_stable_minhash(ai_training_list, ai_testing_list, non_ai_testing_list, k, p):
    # Generate hash functions for the training set
    hash_functions = generate_p_stable_hash_functions(k, p)

    # Calculate pairwise similarity for the training list
    training_signatures = [minhash_signature(train_string, hash_functions, p) for train_string in ai_training_list]
    original_threshold = np.mean([jaccard_similarity(sign1, sign2) for sign1 in training_signatures for sign2 in training_signatures])
    
    threshold_percentages = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

    for threshold_percentage in threshold_percentages:

        threshold = original_threshold * threshold_percentage

        y_values = []

        correct_ai = 0
        # Calculate average similarity for each string in the testing list
        for test_string in ai_testing_list:
            test_signature = minhash_signature(test_string, hash_functions, p)
            average_similarity = np.mean([jaccard_similarity(test_signature, train_signature) for train_signature in training_signatures])
            y_values.append((average_similarity, True))

            # Check if average similarity is above the threshold
            if average_similarity >= threshold:
                correct_ai += 1

        ai_accuracy = correct_ai / len(ai_testing_list)

        correct_non_ai = 0

        for test_string in non_ai_testing_list:
            test_signature = minhash_signature(test_string, hash_functions, p)
            average_similarity = np.mean([jaccard_similarity(test_signature, train_signature) for train_signature in training_signatures])
            y_values.append((average_similarity, False))

            # Check if average similarity is above the threshold
            if average_similarity < threshold:
                correct_non_ai += 1

            
        non_ai_accuracy = correct_non_ai / len(non_ai_testing_list)

        print(f"Results for n={4}, p={threshold_percentage}, t={original_threshold}, p*t={threshold}")
        print(f"Fraction of non-AI abstracts correctly flagged: {non_ai_accuracy}")
        print(f"Fraction of AI abstracts correctly flagged: {ai_accuracy}")

        accuracy = (non_ai_accuracy + ai_accuracy) / 2

        print(f"Accuracy: {accuracy}")
        print("\n")

        plot_scatter(y_values, threshold, 4, accuracy, f'ngram/{4}_gram_tokens_{threshold_percentage}_threshold.png')
        
def plot_scatter(y_values, y_threshold, n, accuracy, save_path=None):
    x_count = len(y_values)
    random.shuffle(y_values)

    plt.figure(figsize=(10, 6))  

    # Plotting the scatter plot
    x_values = range(1, x_count + 1)
    colors = ['green' if not label else 'red' for _, label in y_values]

    plt.scatter(x_values, [y[0] for y in y_values], c=colors, s=10)  # Adjust the 's' parameter as needed
    plt.axhline(y=y_threshold, color='blue', linestyle='--', label=f'Threshold: {y_threshold:.4f}')

    # Display Accuracy near the top-right corner
    plt.text(0.95, 0.95, f'Accuracy: {accuracy:.4f}', transform=plt.gca().transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')

    plt.xlabel('Abstract')
    plt.ylabel('Jaccard Similarity')
    plt.title(f'Scatter Plot for {n}-gram Minhash')

    # Place the legend outside the actual graph on the top right
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Human'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='AI'),
        plt.Line2D([0], [0], color='blue', linestyle='--', label=f'Threshold: {y_threshold:.4f}')
    ], loc='upper right', bbox_to_anchor=(1.25, 0.85))

    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()


def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"
    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"

    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)
    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts_test = read_lines_from_txt(non_ai_generated_file)

    k = 100  # Number of hash functions
    p = 2**31 - 1  # A large prime number for hash function

    p_stable_minhash(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts_test, k, p)


if __name__ == "__main__":
    main()
