import csv
import os

from sklearn.model_selection import train_test_split


def parse_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header
        for row in reader:
            title, abstract, update_date, ai_generated = row
            data.append({
                'title': title.strip(),
                'abstract': abstract.strip().replace("\n", " "),
                'update_date': update_date.strip(),
                'ai_generated': ai_generated.strip() == 'True'
            })
    return data


def main(directory_path):
    ai_generated_set = set()
    non_ai_generated_set = set()

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            parsed_data = parse_csv(file_path)
            
            for entry in parsed_data:

                abstract = entry['abstract']

                if entry['ai_generated']:
                    ai_generated_set.add(abstract)
                else:
                    non_ai_generated_set.add(abstract)
                    
    # Split the ai_generated_set into training (80%) and testing (20%) sets
    ai_generated_list = list(ai_generated_set)
    ai_train_set, ai_test_set = train_test_split(ai_generated_list, test_size=0.2, random_state=42)

    with open('ai_generated_abstracts_train.txt', 'w', encoding='utf-8') as ai_train_file:
        ai_train_file.writelines(abstract + '\n' for abstract in ai_train_set)

    with open('ai_generated_abstracts_test.txt', 'w', encoding='utf-8') as ai_test_file:
        ai_test_file.writelines(abstract + '\n' for abstract in ai_test_set)

    with open('non_ai_generated_abstracts.txt', 'w', encoding='utf-8') as non_ai_file:
        non_ai_file.writelines(abstract + '\n' for abstract in non_ai_generated_set)


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, '..', 'data')
    main(data_directory)
