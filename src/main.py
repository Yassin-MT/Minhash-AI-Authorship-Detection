from datasketch import MinHash
from simhash import Simhash

import numpy as np
import matplotlib.pyplot as plt


def compute_minhash(text, num_perm=128, n=3):
    minhash = MinHash(num_perm=num_perm)

    # Generate n-gram tokens
    tokens = [text[i:i+n] for i in range(len(text) - n + 1)]

    for token in tokens:
        minhash.update(token.encode('utf-8'))

    return minhash

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


    return is_not_similar

def compare_ai_to_ai(ai_minhash_test, ai_minhashes, ai_threshold):
    # Compute Jaccard similarity between the AI and each AI abstract
    similarities = [calculate_jaccard_similarity(ai_minhash_test, ai_minhash) for ai_minhash in ai_minhashes]

    average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    is_similar = average_similarity >= ai_threshold

    # if is_similar:
    #     print(f"The AI abstract is similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {ai_threshold})")
    # else:
    #     print(f"The AI abstract is not similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {ai_threshold})")

    return is_similar

def preprocess_abstracts(abstracts, n=3):
    # Compute MinHash for each abstract
    return [compute_minhash(abstract, n=n) for abstract in abstracts]



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

    k_values = [5, 10, 15, 20, 25]  # Example: Test for K-shingles with sizes 5, 10, 15

    for k in k_values:
        ai_minhashes_kshingle = preprocess_abstracts_kshingle(ai_abstracts_train, k=k)
        ai_threshold_kshingle = average_jaccard_similarity(ai_minhashes_kshingle)

        correct_flags_count_non_ai_kshingle = 0

        for non_ai_abstract in non_ai_abstracts:
            non_ai_minhash_kshingle = compute_minhash_kshingle(non_ai_abstract, k=k)
            is_correct_flag = compare_non_ai_to_ai(non_ai_minhash_kshingle, ai_minhashes_kshingle, ai_threshold_kshingle)

            # Increment the count if the flag is correct
            if is_correct_flag:
                correct_flags_count_non_ai_kshingle += 1

        # Calculate the fraction of correct flags for non-AI abstracts
        total_non_ai_abstracts = len(non_ai_abstracts)
        fraction_correct_flags_non_ai_kshingle = correct_flags_count_non_ai_kshingle / total_non_ai_abstracts

        correct_flags_count_ai_kshingle = 0

        for ai_abstract_test in ai_abstracts_test:
            ai_minhash_test_kshingle = compute_minhash_kshingle(ai_abstract_test, k=k)
            is_correct_flag = compare_ai_to_ai(ai_minhash_test_kshingle, ai_minhashes_kshingle, ai_threshold_kshingle)

            # Increment the count if the flag is correct
            if is_correct_flag:
                correct_flags_count_ai_kshingle += 1

        # Calculate the fraction of correct flags for AI abstracts
        total_ai_abstracts_test = len(ai_abstracts_test)
        fraction_correct_flags_ai_kshingle = correct_flags_count_ai_kshingle / total_ai_abstracts_test

        # Calculate the overall fraction of correct flags
        fraction_correct_flags_kshingle = (correct_flags_count_ai_kshingle + correct_flags_count_non_ai_kshingle) / (
                    total_ai_abstracts_test + total_non_ai_abstracts)

        print(f"\nResults for K-shingles with k={k}:")
        print(f"Fraction of non-AI abstracts correctly flagged: {fraction_correct_flags_non_ai_kshingle}")
        print(f"Fraction of AI abstracts correctly flagged: {fraction_correct_flags_ai_kshingle}")
        print(f"Fraction of abstracts correctly flagged: {fraction_correct_flags_kshingle}")
        print(
            f"Fraction of abstracts correctly flagged (normalized): {(fraction_correct_flags_ai_kshingle + fraction_correct_flags_non_ai_kshingle) / 2}")


def ngram(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts):
    n_gram_values = [2, 3, 4, 5, 6] 

    for n in n_gram_values:
        print(f"\nTesting for {n}-gram tokens:")

        ai_minhashes = preprocess_abstracts(ai_abstracts_train, n=n)
        ai_threshold = average_jaccard_similarity(ai_minhashes)

        correct_flags_count_non_ai = 0

        for non_ai_abstract in non_ai_abstracts:
            non_ai_minhash = compute_minhash(non_ai_abstract, n=n)
            is_correct_flag = compare_non_ai_to_ai(non_ai_minhash, ai_minhashes, ai_threshold)

            # Increment the count if the flag is correct
            if is_correct_flag:
                correct_flags_count_non_ai += 1

        # Calculate the fraction of correct flags
        total_non_ai_abstracts = len(non_ai_abstracts)
        fraction_correct_flags_non_ai = correct_flags_count_non_ai / total_non_ai_abstracts

        correct_flags_count_ai = 0

        for ai_abstract_test in ai_abstracts_test:
            ai_minhash_test = compute_minhash(ai_abstract_test, n=n)
            is_correct_flag = compare_ai_to_ai(ai_minhash_test, ai_minhashes, ai_threshold)

            # Increment the count if the flag is correct
            if is_correct_flag:
                correct_flags_count_ai += 1

        # Calculate the fraction of correct flags
        total_ai_abstracts_test = len(ai_abstracts_test)
        fraction_correct_flags_ai = correct_flags_count_ai / total_ai_abstracts_test

        fraction_correct_flags = (correct_flags_count_ai + correct_flags_count_non_ai) / (total_ai_abstracts_test + total_non_ai_abstracts)

        print(f"Fraction of non-AI abstracts correctly flagged: {fraction_correct_flags_non_ai}")
        print(f"Fraction of AI abstracts correctly flagged: {fraction_correct_flags_ai}")

        print(f"Fraction of abstracts correctly flagged: {fraction_correct_flags}")
        print(f"Fraction of abstracts correctly flagged (normalized): {(fraction_correct_flags_ai + fraction_correct_flags_non_ai) / 2}")




# def plot_results(threshold, ai_abstracts, non_ai_abstracts, is_ai_generated):
#     x_values = range(len(ai_abstracts) + len(non_ai_abstracts))
#     y_values = [threshold] * len(x_values)

#     plt.plot(x_values, y_values, 'r--', label='Threshold')

#     colors = ['green' if generated else 'red' for generated in is_ai_generated]

#     plt.scatter(x_values, [threshold] * len(x_values), c=colors, label='Abstracts')

#     plt.xlabel('Abstracts')
#     plt.ylabel('Jaccard Similarity')
#     plt.title('Jaccard Similarity Comparison')
#     plt.legend()
#     plt.show()


def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"

    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"


    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)

    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts = read_lines_from_txt(non_ai_generated_file)

    ngram(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts)
    # kshingle(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts)
   





if __name__ == "__main__":
    main()

