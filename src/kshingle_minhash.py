from datasketch import MinHash
from simhash import Simhash

import numpy as np
import matplotlib.pyplot as plt
import random

def calculate_jaccard_similarity(minhash1, minhash2):
    # Compute the Jaccard similarity between two MinHash objects
    return minhash1.jaccard(minhash2)

def average_jaccard_similarity(minhash_list):

    # Compute pairwise Jaccard similarities
    pairwise_similarities = []
    for i in range(len(minhash_list)):
        for j in range(i + 1, len(minhash_list)):
            similarity = calculate_jaccard_similarity(minhash_list[i], minhash_list[j])
            pairwise_similarities.append(similarity)

    # Calculate the average similarity
    if pairwise_similarities:
        average_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
        return average_similarity
    else:
        return 0.0


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

def compare_non_ai_to_ai(non_ai_minhash, ai_minhashes, ai_threshold):
    # Compute Jaccard similarity between the non-AI and each AI abstract
    similarities = [calculate_jaccard_similarity(non_ai_minhash, ai_minhash) for ai_minhash in ai_minhashes]

    # Calculate the average similarity
    average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Compare average similarity to the AI threshold
    is_not_similar = average_similarity < ai_threshold

    # if is_not_similar:
    #     print(f"The non-AI abstract is not similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {ai_threshold})")
    # else:
    #     print(f"The non-AI abstract is similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {ai_threshold})")


    return is_not_similar, average_similarity

def compare_ai_to_ai(ai_minhash_test, ai_minhashes, ai_threshold):
    # Compute Jaccard similarity between the AI and each AI abstract
    similarities = [calculate_jaccard_similarity(ai_minhash_test, ai_minhash) for ai_minhash in ai_minhashes]

    average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    is_similar = average_similarity >= ai_threshold

    # if is_similar:
    #     print(f"The AI abstract is similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {ai_threshold})")
    # else:
    #     print(f"The AI abstract is not similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {ai_threshold})")

    return is_similar, average_similarity


def compute_minhash_kshingle(text, num_perm=128, k=5):
    minhash = MinHash(num_perm=num_perm)

    # Generate k-shingles
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        shingles.add(shingle)

    for shingle in shingles:
        minhash.update(shingle.encode('utf-8'))

    return minhash

def preprocess_abstracts_kshingle(abstracts, k=5):
    # Compute MinHash for each abstract using K-shingling
    return [compute_minhash_kshingle(abstract, k=k) for abstract in abstracts]

def kshingle(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts):

    k_values = [10, 15, 20, 25, 30]  # Example: Test for K-shingles with sizes 5, 10, 15
    y_values_matrix = []

    for k in k_values:
        print(f"\nTesting for {k}-shingle tokens:")

        ai_minhashes_kshingle = preprocess_abstracts_kshingle(ai_abstracts_train, k=k)
        original_ai_threshold_kshingle = average_jaccard_similarity(ai_minhashes_kshingle)

        threshold_reductions = [0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.005]

        x_values_line_graph = []
        y_values_line_graph = []

        for threshold_reduction in threshold_reductions:
            x_values_line_graph.append(threshold_reduction)

            y_values = []
            x_count = 0
            
            ai_threshold_kshingle = original_ai_threshold_kshingle * threshold_reduction

            correct_flags_count_non_ai_kshingle = 0

            for non_ai_abstract in non_ai_abstracts:
                non_ai_minhash_kshingle = compute_minhash_kshingle(non_ai_abstract, k=k)
                is_correct_flag, similarity = compare_non_ai_to_ai(non_ai_minhash_kshingle, ai_minhashes_kshingle, ai_threshold_kshingle)
                
                x_count += 1

                y_values.append((similarity, False))

                # Increment the count if the flag is correct
                if is_correct_flag:
                    correct_flags_count_non_ai_kshingle += 1

            # Calculate the fraction of correct flags for non-AI abstracts
            total_non_ai_abstracts = len(non_ai_abstracts)
            fraction_correct_flags_non_ai_kshingle = correct_flags_count_non_ai_kshingle / total_non_ai_abstracts

            correct_flags_count_ai_kshingle = 0

            for ai_abstract_test in ai_abstracts_test:
                ai_minhash_test_kshingle = compute_minhash_kshingle(ai_abstract_test, k=k)
                is_correct_flag, similarity = compare_ai_to_ai(ai_minhash_test_kshingle, ai_minhashes_kshingle, ai_threshold_kshingle)

                x_count += 1

                y_values.append((similarity, True))

                # Increment the count if the flag is correct
                if is_correct_flag:
                    correct_flags_count_ai_kshingle += 1

            # Calculate the fraction of correct flags for AI abstracts
            total_ai_abstracts_test = len(ai_abstracts_test)
            fraction_correct_flags_ai_kshingle = correct_flags_count_ai_kshingle / total_ai_abstracts_test

            print(f"\nResults for K-shingles with k={k}, and {threshold_reduction * 100}% of threshold:")
            print(f"Threshold: {ai_threshold_kshingle}")

            print(f"Fraction of non-AI abstracts correctly flagged: {fraction_correct_flags_non_ai_kshingle}")
            print(f"Fraction of AI abstracts correctly flagged: {fraction_correct_flags_ai_kshingle}")

            normalized = (fraction_correct_flags_ai_kshingle + fraction_correct_flags_non_ai_kshingle) / 2

            print(f"Accuracy: {normalized}")
            
            plot_scatter(y_values, ai_threshold_kshingle, k, normalized, f'kshingle/{k}_shingle_tokens_{threshold_reduction}_threshold.png')

            y_values_line_graph.append(normalized * 100)

        y_values_matrix.append(y_values_line_graph)
    plot_line_graph(x_values_line_graph, y_values_matrix, k_values, 'kshingle/accuracy_over_threshold_percentage.png')




def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"

    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"


    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)

    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts = read_lines_from_txt(non_ai_generated_file)

    kshingle(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts)
   

def plot_scatter(y_values, y_threshold, n, accuracy, save_path=None):
    x_count = len(y_values)

    random.shuffle(y_values)


    # Plotting the scatter plot
    x_values = range(1, x_count + 1)
    colors = ['green' if not label else 'red' for _, label in y_values]

    plt.scatter(x_values, [y[0] for y in y_values], c=colors, s=10)  # Adjust the 's' parameter as needed
    plt.axhline(y=y_threshold, color='blue', linestyle='--', label=f'Threshold: {y_threshold}')
    plt.xlabel('Abstract Index')
    plt.ylabel('Jaccard Similarity')
    plt.title(f'Scatter Plot for {n}-shingle tokens')
    plt.legend()

    # plt.text(0.5, 0.95, f'Accuracy: {accuracy:.2f}%', ha='center', va='center', transform=plt.gca().transAxes,
    #          bbox=dict(facecolor='white', alpha=0.5))

     # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Clear the current figure to avoid overlapping with subsequent figures
    plt.clf()

def plot_line_graph(x_values, y_values, n_gram_values, save_path=None):
    for i, n in enumerate(n_gram_values):
        plt.plot(x_values, y_values[i], label=f'{n}-shingle')

    plt.xlabel('Threshold Percentage')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Threshold Percentage for Different k-shingle Models')
    plt.legend()

    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Clear the current figure to avoid overlapping with subsequent figures
    plt.clf()



if __name__ == "__main__":
    main()

