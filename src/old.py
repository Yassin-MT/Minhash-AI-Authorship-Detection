
# def calculate_average_similarity(simhash_list):
#     # Assuming simhash objects have a length attribute
#     pairwise_distances = [simhash1.distance(simhash2) for simhash1 in simhash_list for simhash2 in simhash_list if simhash1 != simhash2]
#     average_distance = sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0.0
#     return average_distance
    
# def compute_simhash(text):
#     # Create a Simhash object for the given text
#     return Simhash(text)

# def calculate_similarity(simhash1, simhash2):
#     # Compute the Hamming distance between two Simhash objects
#     return simhash1.distance(simhash2)

# def preprocess_abstracts_simhash(abstracts):
#     # Compute Simhash for each abstract
#     return [compute_simhash(abstract) for abstract in abstracts]


# def simhash(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts):

#     ai_simhashes = preprocess_abstracts_simhash(ai_abstracts_train)

#     # Calculate the average similarity for determining the threshold
#     average_similarity_threshold = calculate_average_similarity(ai_simhashes)

#     print(f"Average Similarity Threshold: {average_similarity_threshold}")

#     correct_flags_count_non_ai_simhash = 0

#     for non_ai_abstract in non_ai_abstracts:
#         non_ai_simhash = compute_simhash(non_ai_abstract)
#         is_correct_flag = compare_non_ai_to_ai_simhash(non_ai_simhash, ai_simhashes, average_similarity_threshold)

#         # Increment the count if the flag is correct
#         if is_correct_flag:
#             correct_flags_count_non_ai_simhash += 1

#     # Calculate the fraction of correct flags for non-AI abstracts
#     total_non_ai_abstracts = len(non_ai_abstracts)
#     fraction_correct_flags_non_ai_simhash = correct_flags_count_non_ai_simhash / total_non_ai_abstracts

#     correct_flags_count_ai_simhash = 0

#     for ai_abstract_test in ai_abstracts_test:
#         ai_simhash_test = compute_simhash(ai_abstract_test)
#         is_correct_flag = compare_ai_to_ai_simhash(ai_simhash_test, ai_simhashes, average_similarity_threshold)

#         # Increment the count if the flag is correct
#         if is_correct_flag:
#             correct_flags_count_ai_simhash += 1

#     # Calculate the fraction of correct flags for AI abstracts
#     total_ai_abstracts_test = len(ai_abstracts_test)
#     fraction_correct_flags_ai_simhash = correct_flags_count_ai_simhash / total_ai_abstracts_test

#     # Calculate the overall fraction of correct flags
#     fraction_correct_flags_simhash = (
#         correct_flags_count_ai_simhash + correct_flags_count_non_ai_simhash
#     ) / (total_ai_abstracts_test + total_non_ai_abstracts)

#     print("\nResults for Simhash:")
#     print(f"Fraction of non-AI abstracts correctly flagged: {fraction_correct_flags_non_ai_simhash}")
#     print(f"Fraction of AI abstracts correctly flagged: {fraction_correct_flags_ai_simhash}")
#     print(f"Fraction of abstracts correctly flagged: {fraction_correct_flags_simhash}")
#     print(f"Fraction of abstracts correctly flagged (normalized): {(fraction_correct_flags_ai_simhash + fraction_correct_flags_non_ai_simhash) / 2}")


# def compare_non_ai_to_ai_simhash(non_ai_simhash, ai_simhashes, similarity_threshold):
#     # Compare the Simhash of non-AI abstract to the AI abstracts
#     similarities = [non_ai_simhash.distance(ai_simhash) for ai_simhash in ai_simhashes]

#     # Calculate the average similarity
#     average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

#     # Compare the average similarity to a threshold
#     return average_similarity >= similarity_threshold


# def compare_ai_to_ai_simhash(ai_simhash_test, ai_simhashes, similarity_threshold):
#     # Compare the Simhash of AI abstract to the AI abstracts
#     similarities = [ai_simhash_test.distance(ai_simhash) for ai_simhash in ai_simhashes]

#     # Calculate the average similarity
#     average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

#     # Compare the average similarity to a threshold
#     return average_similarity < similarity_threshold


# def calculate_jaccard_similarity_projection_matrix(minhash1, minhash2, num_hashes):
#     # Compute the Jaccard similarity between two MinHash objects
#     return np.sum(minhash1 == minhash2) / num_hashes

# def compute_minhash_projection_matrix(projection_matrix, text, num_hashes=100, num_perm=128, n=3):
#     minhash = np.inf * np.ones(num_hashes)

#     # Generate n-gram tokens
#     tokens = [text[i:i + n] for i in range(len(text) - n + 1)]

#     for token in tokens:
#         # Apply Cauchy distribution for each hash function
#         hash_values = np.array([hash(token + str(i)) % num_perm for i in range(num_hashes)])
        
#         # Transpose projection_matrix
#         transposed_projection_matrix = np.transpose(projection_matrix)
        
#         minhash = np.minimum(minhash, (hash_values + transposed_projection_matrix) % num_perm)

#     return minhash


# def calculate_average_similarity_projection_matrix(minhash_list, num_hashes):
#     # Compute pairwise similarities
#     pairwise_similarities = []
#     for i in range(len(minhash_list)):
#         for j in range(i + 1, len(minhash_list)):
#             similarity = calculate_jaccard_similarity_projection_matrix(minhash_list[i], minhash_list[j], num_hashes)
#             # Ensure that similarity values are between 0 and 1
#             similarity = max(0, min(1, similarity))
#             pairwise_similarities.append(similarity)

#     # Calculate the average similarity
#     if pairwise_similarities:
#         average_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
#         return average_similarity
#     else:
#         return 0.0


# def preprocess_abstracts_projection_matrix(abstracts, projection_matrix, num_hashes=100, num_perm=128, n=3):
#     # Compute MinHash for each abstract
#     return [compute_minhash_projection_matrix(projection_matrix, abstract, num_hashes=num_hashes, num_perm=num_perm, n=n) for abstract in abstracts]

# def compare_non_ai_to_ai_projection_matrix(non_ai_minhash, ai_minhashes, similarity_threshold):
#     # Compare the MinHash of non-AI abstract to the AI abstracts
#     similarities = [calculate_jaccard_similarity_projection_matrix(non_ai_minhash, ai_minhash, len(non_ai_minhash)) for ai_minhash in ai_minhashes]

#     # Calculate the average similarity
#     average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

#     # Compare the average similarity to a threshold
#     is_not_similar = average_similarity < similarity_threshold

#     if is_not_similar:
#         print(f"The non-AI abstract is not similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {similarity_threshold})")
#     else:
#         print(f"The non-AI abstract is similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {similarity_threshold})")

#     return is_not_similar

# def compare_ai_to_ai_projection_matrix(ai_minhash_test, ai_minhashes, similarity_threshold):
#     # Compare the MinHash of AI abstract to the AI abstracts
#     similarities = [calculate_jaccard_similarity_projection_matrix(ai_minhash_test, ai_minhash, len(ai_minhash_test)) for ai_minhash in ai_minhashes]

#     # Calculate the average similarity
#     average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

#     # Compare the average similarity to a threshold
#     is_similar =  average_similarity >= similarity_threshold

#     if is_similar:
#         print(f"The AI abstract is similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {similarity_threshold})")
#     else:
#         print(f"The AI abstract is not similar to the AI abstracts with an average similarity of {average_similarity} (threshold: {similarity_threshold})")

#     return is_similar

    

# def ngram_projection_matrix(ai_abstracts_train, ai_abstracts_test, non_ai_abstracts):
#     n_gram_values = [2, 3, 4, 5, 6] 

#     # Parameters
#     num_perm = 128  # Number of hash functions
#     num_hashes = 100  # Number of hashes for each hash function
#     n = 3  # N-gram size for tokenization
#     cauchy_scale = 0.5  # Cauchy distribution scale parameter

#     # Generate random projection matrix from Cauchy distribution
#     projection_matrix = np.random.standard_cauchy(size=(num_hashes, num_perm))

#     for n in n_gram_values:
#         print(f"\nTesting for {n}-gram tokens:")

#         ai_minhashes = preprocess_abstracts_projection_matrix(ai_abstracts_train, projection_matrix, num_hashes=num_hashes, num_perm=num_perm, n=n)
#         ai_threshold = calculate_average_similarity_projection_matrix(ai_minhashes, num_hashes)

#         correct_flags_count_non_ai = 0

#         for non_ai_abstract in non_ai_abstracts:
#             non_ai_minhash = compute_minhash_projection_matrix(projection_matrix, non_ai_abstract, num_hashes=num_hashes, num_perm=num_perm, n=n)
#             is_correct_flag = compare_non_ai_to_ai_projection_matrix(non_ai_minhash, ai_minhashes, ai_threshold)

#             # Increment the count if the flag is correct
#             if is_correct_flag:
#                 correct_flags_count_non_ai += 1

#         # Calculate the fraction of correct flags
#         total_non_ai_abstracts = len(non_ai_abstracts)
#         fraction_correct_flags_non_ai = correct_flags_count_non_ai / total_non_ai_abstracts

#         correct_flags_count_ai = 0

#         for ai_abstract_test in ai_abstracts_test:
#             ai_minhash_test = compute_minhash_projection_matrix(projection_matrix, ai_abstract_test, num_hashes=num_hashes, num_perm=num_perm, n=n)
#             is_correct_flag = compare_ai_to_ai_projection_matrix(ai_minhash_test, ai_minhashes, ai_threshold)

#             # Increment the count if the flag is correct
#             if is_correct_flag:
#                 correct_flags_count_ai += 1

#         # Calculate the fraction of correct flags
#         total_ai_abstracts_test = len(ai_abstracts_test)
#         fraction_correct_flags_ai = correct_flags_count_ai / total_ai_abstracts_test

#         fraction_correct_flags = (correct_flags_count_ai + correct_flags_count_non_ai) / (total_ai_abstracts_test + total_non_ai_abstracts)

#         print(f"Fraction of non-AI abstracts correctly flagged: {fraction_correct_flags_non_ai}")
#         print(f"Fraction of AI abstracts correctly flagged: {fraction_correct_flags_ai}")

#         print(f"Fraction of abstracts correctly flagged: {fraction_correct_flags}")
#         print(f"Fraction of abstracts correctly flagged (normalized): {(fraction_correct_flags_ai + fraction_correct_flags_non_ai) / 2}")



