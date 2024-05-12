"""
You should implement the functions in this file.
The goal is to provide you with the basic structure.
Feel free to create additional functions.
You are not limited to a single blocking or matching function.
"""
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re

def generate_blocking_key(row: pd.Series):

    # Ensure the field exists and is not NaN, then apply regex to the whole string
    if pd.notna(row[1]):
        # Find all matches of the pattern in the entire string
        matches = re.findall(r"\w+\s\w+\d+", str(row[1]))
        # If matches are found, take the first match and then the first three letters of it
        if matches:
            name_key = matches[0].lower() 
        else:
            name_key = ''  # If no matches, set name_key to empty string
    else:
        name_key = ''  # Handle NaN values

    # Prepare the blocking key
    all_keys = [name_key]  # Only one key considered here
    all_keys = [key for key in all_keys if key]  # Filter out empty keys

    if not all_keys:
        return ''  # If no valid keys, return empty string

    # Combine all keys into a single string, sorted and de-duplicated
    blocking_key = " ".join(sorted(set(all_keys)))
    return blocking_key

def create_blocks(df: pd.DataFrame):
    '''
    This function creates maps prodicts into blocks based on the blocking key
    :param df: the input data containing the information about the products
    :return: a dictionary that the key is the generated blocking key and the value is the product id
    '''

    blocks = defaultdict(list)
    for rowid in tqdm(range(df.shape[0])):
        blocking_key = generate_blocking_key(df.iloc[rowid,:])
        if blocking_key != '':
            blocks[blocking_key].append(rowid)

    return blocks


def generate_matches(blocks: defaultdict, df: pd.DataFrame):
    '''
    based on the created blocks, this function generates candidate matches
    :param blocks: the blocks where the key represents the blocking key and the value is the product ids
    :param df: the input data containing the information about the products
    :return: a list of tuples, where each tuple is a pair of product id, that are returned as candidate matches
    '''

    candidate_pairs = []
    for key in tqdm(blocks):
        row_ids = list(sorted(blocks[key]))
        if len(row_ids) < 100:  # skip keys that are too common
            for i in range(len(row_ids)):
                for j in range(i + 1, len(row_ids)):
                    candidate_pairs.append((row_ids[i], row_ids[j]))

    similarity_threshold = .5
    jaccard_similarities = []
    candidate_pairs_product_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # get product ids
        product_id1 = df['id'][id1]
        product_id2 = df['id'][id2]
        if product_id1 < product_id2:  # NOTE: This is to make sure in the final candidates, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
            candidate_pairs_product_ids.append((product_id1, product_id2))
        else:
            candidate_pairs_product_ids.append((product_id2, product_id1))

        # compute jaccard similarity
        name1 = str(df[df.columns.values[1]][id1])
        name2 = str(df[df.columns.values[1]][id2])
        s1 = set(name1.split())
        s2 = set(name2.split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_product_ids = [x for s, x in sorted(zip(jaccard_similarities, candidate_pairs_product_ids), reverse=True) if s >= similarity_threshold]
    return candidate_pairs_product_ids


def evaluate(match_pairs: list, ground_truth: pd.DataFrame):
    '''
    Calculates the precision, recall, and F1 score
    :param candidate_pairs: list of candidate pairs
    :param ground_truth: the dataframe containing the actual matches
    :return: Does not return anything
    '''
    gt = list(zip(ground_truth['lid'], ground_truth['rid']))
    tp = len(set(match_pairs).intersection(set(gt)))
    fp = len(set(match_pairs).difference(set(gt)))
    fn = len(set(gt).difference(set(match_pairs)))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    print(f'Reported # of Pairs: {len(match_pairs)}')
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    print('FN: {}'.format(fn))
    print('Recal: {}'.format(recall))
    print('Precision: {}'.format(precision))
    print('F1: {}'.format((2 * precision * recall) / (precision + recall)))


# read the datasets
# Z1 = pd.read_csv("Data/Z1.csv")
Z2 = pd.read_csv("Data/Z2.csv")

starting_time = time.time()
# perform blocking
# blocks_Z1 = Z1_candidate_pairs = create_blocks(Z1)
blocks_Z2 = Z2_candidate_pairs = create_blocks(Z2)

# Generate candidates
# candidates_Z1 = generate_matches(blocks_Z1, Z1)
candidates_Z2 = generate_matches(blocks_Z2, Z2)
ending_time = time.time()

# Evaluation
print('------------- Evaluation Results --------------')
print(f'Runtime: {ending_time - starting_time} Seconds')
# print('------------- First Dataset --------------')
# evaluate(candidates_Z1, pd.read_csv('Data/ZY1.csv'))
print('------------- Second Dataset --------------')
evaluate(candidates_Z2, pd.read_csv('Data/ZY2.csv'))