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


def p_stable_hash(x, a, b, p, max_val):
    result = int(((a * x + b) % p) * max_val)
    return result

def generate_p_stable_hash_functions(k, p, max_val):
    return [(np.random.randint(1, p), np.random.randint(0, p)) for _ in range(k)]

def minhash_signature(string, hash_functions, p, max_val, n=6):
    signature = np.inf * np.ones(len(hash_functions), dtype=int)
    for i, hash_func_params in enumerate(hash_functions):
        for j in range(len(string) - n + 1):
            token = string[j:j+n]

            hash_value = p_stable_hash(hash(token), *hash_func_params, p, max_val)
            signature[i] = min(signature[i], hash_value)
    return signature

def jaccard_similarity(signature1, signature2):
    intersection_size = np.sum(signature1 == signature2)
    union_size = len(signature1)
    return intersection_size / union_size

def average_pairwise_similarity(strings, k, p, max_val):
    hash_functions = generate_p_stable_hash_functions(k, p, max_val)
    signatures = [minhash_signature(string, hash_functions, p, max_val) for string in strings]
    
    total_similarity = 0
    total_pairs = 0

    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            similarity = jaccard_similarity(signatures[i], signatures[j])
            total_similarity += similarity
            total_pairs += 1

    average_similarity = total_similarity / total_pairs
    return average_similarity

def p_stable_minhash(ai_training_list, ai_testing_list, non_ai_testing_list, k, p, max_val):
    # Generate hash functions for the training set
    hash_functions = generate_p_stable_hash_functions(k, p, max_val)

    # Calculate pairwise similarity for the training list
    training_signatures = [minhash_signature(train_string, hash_functions, p, max_val) for train_string in ai_training_list]
    threshold = np.mean([jaccard_similarity(sign1, sign2) for sign1 in training_signatures for sign2 in training_signatures])
    threshold = threshold * 0.7


    y_values = []

    total_ai = 0
    correct_ai = 0
    # Calculate average similarity for each string in the testing list
    for test_string in ai_testing_list:
        test_signature = minhash_signature(test_string, hash_functions, p, max_val)
        average_similarity = np.mean([jaccard_similarity(test_signature, train_signature) for train_signature in training_signatures])
        y_values.append((average_similarity, True))

        # Check if average similarity is above the threshold
        if average_similarity < threshold:
            print(f"String is not similar to the training set.")
        else:
            print(f"String is similar to the training set.")
            correct_ai += 1

        
        total_ai += 1

    total_non_ai = 0
    correct_non_ai = 0

    for test_string in non_ai_testing_list:
        test_signature = minhash_signature(test_string, hash_functions, p, max_val)
        average_similarity = np.mean([jaccard_similarity(test_signature, train_signature) for train_signature in training_signatures])
        y_values.append((average_similarity, False))

        # Check if average similarity is above the threshold
        if average_similarity < threshold:
            print(f"String is not similar to the training set.")
            correct_non_ai += 1
        else:
            print(f"String is similar to the training set.")

        
        total_non_ai += 1

    plot_scatter(y_values, threshold, f'pstable/output.png')

    accuracy = ((correct_ai / total_ai) + (correct_non_ai / total_non_ai)) / 2
    print(f"Accuracy: {accuracy}")

        
def plot_scatter(y_values, y_threshold, save_path=None):
    x_count = len(y_values)

    random.shuffle(y_values)

    x_values = range(1, x_count + 1)
    colors = ['green' if not label else 'red' for _, label in y_values]

    plt.scatter(x_values, [y[0] for y in y_values], c=colors, s=10)  # Adjust the 's' parameter as needed
    plt.axhline(y=y_threshold, color='blue', linestyle='--', label=f'Threshold: {y_threshold}')
    plt.xlabel('Abstract Index')
    plt.ylabel('Hamming Distance')
    plt.title(f'Scatter Plot')
    plt.legend()

    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path)


def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"

    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"


    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)

    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts_test = read_lines_from_txt(non_ai_generated_file)
    

    k = 100  # Number of hash functions
    p = 2**31 - 1  # A large prime number for hash function
    max_val = 10000  # Adjust this based on the expected range of hash values

    p_stable_minhash(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts_test, k, p, max_val)

if __name__ == "__main__":
    main()
