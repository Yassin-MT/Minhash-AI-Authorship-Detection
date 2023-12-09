
import matplotlib.pyplot as plt
import random
from simhash import Simhash


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


def hamming_distance(simhash1, simhash2):
    return simhash1.distance(simhash2)


# Function to calculate average pairwise Hamming distance
def average_pairwise_hamming_distance(simhash_list):

    # Calculate pairwise Hamming distances
    pairwise_distances = [
        hamming_distance(simhash1, simhash2)
        for i, simhash1 in enumerate(simhash_list)
        for simhash2 in simhash_list[i + 1:]
    ]

    # Calculate the average pairwise Hamming distance
    average_distance = sum(pairwise_distances) / len(pairwise_distances)
    return average_distance
 


# Function to calculate average Hamming distance for a list of SimHash values
def average_hamming_distance(reference_simhash, simhash_list):
    distances = [hamming_distance(reference_simhash, simhash) for simhash in simhash_list]
    
    if distances:
        average_distance = sum(distances) / len(distances)
        return average_distance

def plot_scatter(y_values, y_threshold, accuracy, save_path=None):
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
    plt.title(f'Scatter Plot for SimHash')

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

    
def simhash(ai_train, ai_test, non_ai_test):
    ai_train_simhashes = [Simhash(string) for string in ai_train]

    threshold = average_pairwise_hamming_distance(ai_train_simhashes)
    y_values = []

    correct_ai = 0

    for ai_test_specific in ai_test:
        average_specific = average_hamming_distance(Simhash(ai_test_specific), ai_train_simhashes)

        y_values.append((average_specific, True))
        if average_specific <= threshold:
            correct_ai += 1
    
    ai_accuracy = correct_ai / len(ai_test)

    correct_non_ai = 0

    for non_ai_test_specific in non_ai_test:
        average_specific = average_hamming_distance(Simhash(non_ai_test_specific), ai_train_simhashes)
        y_values.append((average_specific, False))

        if average_specific > threshold:
            correct_non_ai += 1

    non_ai_accuracy = correct_non_ai / len(non_ai_test)

    print(f"Results for t={threshold}")
    print(f"Fraction of non-AI abstracts correctly flagged: {non_ai_accuracy}")
    print(f"Fraction of AI abstracts correctly flagged: {ai_accuracy}")

    accuracy = (non_ai_accuracy + ai_accuracy) / 2

    print(f"Accuracy: {accuracy}")
    print("\n")

    plot_scatter(y_values, threshold, accuracy, f'simhash/output.png')



def main():
    ai_generated_train_file = "ai_generated_abstracts_train.txt"
    ai_generated_test_file = "ai_generated_abstracts_test.txt"
    non_ai_generated_file = "non_ai_generated_abstracts.txt"

    ai_abstracts_train = read_lines_from_txt(ai_generated_train_file)
    ai_abstracts_test = read_lines_from_txt(ai_generated_test_file)
    non_ai_abstracts_test = read_lines_from_txt(non_ai_generated_file)
    
    simhash(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts_test)


if __name__ == "__main__":
    main()

