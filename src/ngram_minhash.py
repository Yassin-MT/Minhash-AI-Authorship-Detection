from datasketch import MinHash
import matplotlib.pyplot as plt
import random


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
    similarities = [calculate_jaccard_similarity(non_ai_minhash, ai_minhash) for ai_minhash in ai_minhashes]

    # Calculate the average similarity
    average_similarity = sum(similarities) / len(similarities)
    is_not_similar = average_similarity < ai_threshold

    return is_not_similar, average_similarity


def compare_ai_to_ai(ai_minhash_test, ai_minhashes, ai_threshold):
    # Compute Jaccard similarity between the AI and each AI abstract
    similarities = [calculate_jaccard_similarity(ai_minhash_test, ai_minhash) for ai_minhash in ai_minhashes]

    average_similarity = sum(similarities) / len(similarities)
    is_similar = average_similarity >= ai_threshold

    return is_similar, average_similarity


def preprocess_abstracts(abstracts, n=3):
    # Compute MinHash for each abstract
    return [compute_minhash(abstract, n=n) for abstract in abstracts]


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
    


def plot_line_graph(x_values, y_values, n_gram_values, save_path=None):
    plt.figure(figsize=(10, 6))  

    for i, n in enumerate(n_gram_values):
        plt.plot(x_values, y_values[i], label=f'{n}-gram')
    
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xlabel('Threshold Percentage')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold Percentage for Different n-gram Models')
    
    plt.legend(loc='upper right')
    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path)

    plt.close()



def ngram(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts):
    n_gram_values = [4, 5, 6] 

    y_values_matrix = []
    for n in n_gram_values:
        ai_minhashes = preprocess_abstracts(ai_abstracts_train, n=n)
        original_ai_threshold = average_jaccard_similarity(ai_minhashes)

        threshold_percentages = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        
        x_values_line_graph = []
        y_values_line_graph = []

        for threshold_percentage in threshold_percentages:
            x_values_line_graph.append(threshold_percentage)
            y_values = []

            ai_threshold = original_ai_threshold * threshold_percentage

            correct_flags_count_non_ai = 0

            for non_ai_abstract in non_ai_abstracts:
                non_ai_minhash = compute_minhash(non_ai_abstract, n=n)
                is_correct_flag, similarity = compare_non_ai_to_ai(non_ai_minhash, ai_minhashes, ai_threshold)
                y_values.append((similarity, False))

                # Increment the count if the flag is correct
                if is_correct_flag:
                    correct_flags_count_non_ai += 1

            # Calculate the fraction of correct flags
            total_non_ai_abstracts = len(non_ai_abstracts)
            non_ai_accuracy = correct_flags_count_non_ai / total_non_ai_abstracts

            correct_flags_count_ai = 0

            for ai_abstract_test in ai_abstracts_test:
                ai_minhash_test = compute_minhash(ai_abstract_test, n=n)
                is_correct_flag, similarity = compare_ai_to_ai(ai_minhash_test, ai_minhashes, ai_threshold)
                y_values.append((similarity, True))

                # Increment the count if the flag is correct
                if is_correct_flag:
                    correct_flags_count_ai += 1

            # Calculate the fraction of correct flags
            total_ai_abstracts_test = len(ai_abstracts_test)
            ai_accuracy = correct_flags_count_ai / total_ai_abstracts_test

            print(f"Results for n={n}, p={threshold_percentage}, t={original_ai_threshold}, p*t={ai_threshold}")
            print(f"Fraction of non-AI abstracts correctly flagged: {non_ai_accuracy}")
            print(f"Fraction of AI abstracts correctly flagged: {ai_accuracy}")

            accuracy = (non_ai_accuracy + ai_accuracy) / 2

            print(f"Accuracy: {accuracy}")
            print("\n")

            plot_scatter(y_values, ai_threshold, n, accuracy, f'ngram/{n}_gram_tokens_{threshold_percentage}_threshold.png')
            y_values_line_graph.append(accuracy * 100)

        y_values_matrix.append(y_values_line_graph)

    plot_line_graph(x_values_line_graph, y_values_matrix, n_gram_values, 'ngram/accuracy_vs_threshold_percentage.png')


def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"
    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"

    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)
    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts = read_lines_from_txt(non_ai_generated_file)

    ngram(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts)
   

if __name__ == "__main__":
    main()