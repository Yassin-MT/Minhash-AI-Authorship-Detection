
from datasketch import MinHash
import matplotlib.pyplot as plt
import random

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def calculate_jaccard_similarity(minhash1, minhash2):
    # Compute the Jaccard similarity between two MinHash objects
    return minhash1.jaccard(minhash2)

def calculate_cosine_similarity(minhash_A, minhash_B):
    # Estimate Jaccard similarity using MinHash signatures
    jaccard_similarity = minhash_A.jaccard(minhash_B)

    # Convert Jaccard similarity to approximate cosine similarity
    cosine_similarity = jaccard_similarity / (minhash_A.count() * minhash_B.count()) ** 0.5

    return cosine_similarity

def average_jaccard_similarity(minhash_list):
    # Compute pairwise Jaccard similarities
    pairwise_similarities = []
    for i in range(len(minhash_list)):
        for j in range(i + 1, len(minhash_list)):
            similarity = calculate_cosine_similarity(minhash_list[i], minhash_list[j])
            pairwise_similarities.append(similarity)

    # Calculate the average similarity
    average_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
    return average_similarity


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
    similarities = [calculate_cosine_similarity(non_ai_minhash, ai_minhash) for ai_minhash in ai_minhashes]

    # Calculate the average similarity
    average_similarity = sum(similarities) / len(similarities)
    is_not_similar = average_similarity < ai_threshold

    return is_not_similar, average_similarity


def compare_ai_to_ai(ai_minhash_test, ai_minhashes, ai_threshold):
    # Compute Jaccard similarity between the AI and each AI abstract
    similarities = [calculate_cosine_similarity(ai_minhash_test, ai_minhash) for ai_minhash in ai_minhashes]

    average_similarity = sum(similarities) / len(similarities)
    is_similar = average_similarity >= ai_threshold

    return is_similar, average_similarity

def compute_minhash_bert_tokenizer(text, num_perm=128, k=3):
    # Load the BERT tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    # Generate k-token shingles
    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i:i+k])
        shingles.add(shingle)

    minhash = MinHash(num_perm=num_perm)

    for shingle in shingles:
        minhash.update(shingle.encode('utf-8'))

    return minhash

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

def compute_minhash_wordshingle(text, num_perm=128, k=3):
    minhash = MinHash(num_perm=num_perm)
    
    # Tokenize the text into words
    words = text.split()

    # Generate k-word shingles
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i+k])
        shingles.add(shingle)

    for shingle in shingles:
        minhash.update(shingle.encode('utf-8'))

    return minhash

def preprocess_abstracts_kshingle(abstracts, k=5):
    # Compute MinHash for each abstract using K-shingling
    return [compute_minhash_bert_tokenizer(abstract, k=k) for abstract in abstracts]


def kshingle(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts):

    # k_values = [10, 15]
    k_values = [3, 4, 5]

    y_values_matrix = []

    for k in k_values:
        ai_minhashes_kshingle = preprocess_abstracts_kshingle(ai_abstracts_train, k=k)
        original_ai_threshold = average_jaccard_similarity(ai_minhashes_kshingle)

        threshold_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
        # threshold_percentages = [0.4, 0.2, 0.1, 0.05, 0.025, 0.005]

        x_values_line_graph = []
        y_values_line_graph = []

        for threshold_percentage in threshold_percentages:
            x_values_line_graph.append(threshold_percentage)

            y_values = []
            
            ai_threshold = original_ai_threshold * threshold_percentage

            correct_flags_count_non_ai_kshingle = 0

            for non_ai_abstract in non_ai_abstracts:
                non_ai_minhash_kshingle = compute_minhash_bert_tokenizer(non_ai_abstract, k=k)
                is_correct_flag, similarity = compare_non_ai_to_ai(non_ai_minhash_kshingle, ai_minhashes_kshingle, ai_threshold)
                y_values.append((similarity, False))

                # Increment the count if the flag is correct
                if is_correct_flag:
                    correct_flags_count_non_ai_kshingle += 1

            # Calculate the fraction of correct flags for non-AI abstracts
            total_non_ai_abstracts = len(non_ai_abstracts)
            fraction_correct_flags_non_ai_kshingle = correct_flags_count_non_ai_kshingle / total_non_ai_abstracts

            correct_flags_count_ai_kshingle = 0

            for ai_abstract_test in ai_abstracts_test:
                ai_minhash_test_kshingle = compute_minhash_bert_tokenizer(ai_abstract_test, k=k)
                is_correct_flag, similarity = compare_ai_to_ai(ai_minhash_test_kshingle, ai_minhashes_kshingle, ai_threshold)
                y_values.append((similarity, True))

                # Increment the count if the flag is correct
                if is_correct_flag:
                    correct_flags_count_ai_kshingle += 1

            # Calculate the fraction of correct flags for AI abstracts
            total_ai_abstracts_test = len(ai_abstracts_test)
            fraction_correct_flags_ai_kshingle = correct_flags_count_ai_kshingle / total_ai_abstracts_test

            print(f"Results for k={k}, p={threshold_percentage}, t={original_ai_threshold}, p*t={ai_threshold}")

            print(f"Fraction of non-AI abstracts correctly flagged: {fraction_correct_flags_non_ai_kshingle}")
            print(f"Fraction of AI abstracts correctly flagged: {fraction_correct_flags_ai_kshingle}")

            accuracy = (fraction_correct_flags_ai_kshingle + fraction_correct_flags_non_ai_kshingle) / 2

            print(f"Accuracy: {accuracy}")
            
            plot_scatter(y_values, ai_threshold, k, accuracy, f'kshingle_tokenizer_cosine/{k}_shingle_tokenizer_{threshold_percentage}_threshold.png')

            y_values_line_graph.append(accuracy * 100)

        y_values_matrix.append(y_values_line_graph)
    plot_line_graph(x_values_line_graph, y_values_matrix, k_values, 'kshingle_tokenizer_cosine/accuracy_vs_threshold_percentage_shingle_tokenizer.png')


def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"
    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"

    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)
    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts = read_lines_from_txt(non_ai_generated_file)

    kshingle(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts)
   

def plot_scatter(y_values, y_threshold, k, accuracy, save_path=None):
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
    plt.title(f'Scatter Plot for {k}-shingle Minhash')

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
    

def plot_line_graph(x_values, y_values, k_shingle_values, save_path=None):
    plt.figure(figsize=(10, 6))  

    for i, k in enumerate(k_shingle_values):
        plt.plot(x_values, y_values[i], label=f'{k}-shingle')
    
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xlabel('Threshold Percentage')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold Percentage for Different k-shingle Models')
    
    plt.legend(loc='upper right')
    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path)

    plt.close()


if __name__ == "__main__":
    main()

