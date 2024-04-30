import pandas as pd

# implement a function for calculating the jaccard similarity
def calculate_jaccard_similarity(df):
    # Calculate the jaccard similarity using progress_apply for all the rows in the dataframe considering the words
    jaccard_similarities = df.apply(lambda x: len(set(x['text'].split()) & set(x['obfuscatedText'].split())) / len(set(x['text'].split()) | set(x['obfuscatedText'].split())), axis=1)
    return jaccard_similarities

if __name__ == '__main__':
    mech = 'Mhl'
    epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 50]
    for e in epsilons:
        df = pd.read_csv(f'./data/final/SotA/{mech}/obfuscated_Mahalanobis_{e}.csv', sep = ',')
        df['jaccard_similarity'] = calculate_jaccard_similarity(df)
        df.to_csv(f'./data/final/SotA/{mech}/jaccard_similarity_Mahalanobis_{e}.csv', index = False)
        print(f'Mean jaccard similarity for {e}: {df["jaccard_similarity"].mean()}')
        print(f'Finished {e}')
