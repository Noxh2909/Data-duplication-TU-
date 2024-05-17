import pandas as pd
import hashlib
from tqdm import tqdm
import time
from collections import defaultdict
import re

# Constants for column names
NAME_COLUMN_NAME = 1 
NAME_COLUMN_PRICE = 2
NAME_COLUMN_BRAND = 3

def hash_record(record):
    """Create a unique hash for a record based on its content using SHA-256."""
    record_hash = hashlib.sha256(record.encode()).hexdigest()
    return record_hash

def find_and_pair_duplicates(df, column_name):
    """Identify duplicates in the DataFrame based on the specified column and return pairs of their IDs."""
    hash_dict = {}  # Dictionary to store first occurrence of each hash with its ID
    duplicate_pairs = []  # List to store pairs of IDs that are duplicates

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing records"):
        record = row[column_name]
        record_hash = hash_record(record)

        if record_hash in hash_dict:  # If the hash is already in the dictionary, it's a duplicate
            original_id = hash_dict[record_hash]  # Get the ID of the first record with this hash
            duplicate_pairs.append((original_id, row['id']))  # Store the pair of duplicate IDs
        else:
            hash_dict[record_hash] = row['id']  # Store the hash with the ID of the first occurrence

    return duplicate_pairs

def generate_blocking_key(row: pd.Series):
    """Generate a blocking key based on specific patterns in the product name."""
    patterns = [
        r"\b[A-Z0-9]{2,10}-[A-Z0-9]{2,10}\b",  # Specific product codes
        r"\b\w+\b",                            # Any word
    ]
    keys = []
    if pd.notna(row[NAME_COLUMN_NAME]):
        for pattern in patterns:
            matches = re.findall(pattern, str(row[NAME_COLUMN_NAME]), re.IGNORECASE)
            keys.extend([match.lower() for match in matches])
    if not keys:
        return ''
    return ' '.join(sorted(set(keys)))  # Combining and sorting keys 

def create_blocks(df: pd.DataFrame):
    """Map products into blocks based on the blocking key."""
    blocks = defaultdict(list)
    for rowid in tqdm(range(df.shape[0]), desc="Creating blocks"):
        blocking_key = generate_blocking_key(df.iloc[rowid, :])
        if blocking_key != '':
            blocks[blocking_key].append(rowid)
    return blocks

def generate_matches(blocks: defaultdict, df: pd.DataFrame):
    """Generate candidate matches based on the created blocks."""
    candidate_pairs = []
    for key in tqdm(blocks, desc="Generating candidate pairs"):
        row_ids = list(sorted(blocks[key]))
        if len(row_ids) < 100:  # skip keys that are too common
            for i in range(len(row_ids)):
                for j in range(i + 1, len(row_ids)):
                    candidate_pairs.append((row_ids[i], row_ids[j]))

    similarity_threshold = 0.5
    jaccard_similarities = []
    candidate_pairs_product_ids = []
    for it in tqdm(candidate_pairs, desc="Calculating Jaccard similarities"):
        id1, id2 = it

        # get product ids
        product_id1 = df['id'][id1]
        product_id2 = df['id'][id2]
        if product_id1 < product_id2:  # Ensure consistent ordering
            candidate_pairs_product_ids.append((product_id1, product_id2))
        else:
            candidate_pairs_product_ids.append((product_id2, product_id1))

        # compute jaccard similarity
        name1 = str(df[df.columns.values[1]][id1])
        name2 = str(df[df.columns.values[1]][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))

    # Filter candidates by similarity threshold
    candidate_pairs_product_ids = [x for s, x in sorted(zip(jaccard_similarities, candidate_pairs_product_ids), reverse=True) if s >= similarity_threshold]
    return candidate_pairs_product_ids

def evaluate(match_pairs: list, ground_truth: pd.DataFrame):
    """Calculate and print the precision, recall, and F1 score."""
    gt = list(zip(ground_truth['lid'], ground_truth['rid']))
    tp = len(set(match_pairs).intersection(set(gt)))
    fp = len(set(match_pairs).difference(set(gt)))
    fn = len(set(gt).difference(set(match_pairs)))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    print(f'Reported # of Pairs: {len(match_pairs)}')
    print(f'TP: {tp}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1: {(2 * precision * recall) / (precision + recall)}')

# Read the datasets
Z1 = pd.read_csv("Data/Z1.csv")
# Z2 = pd.read_csv("Data/Z2.csv")

starting_time = time.time()

# Perform blocking
blocks_Z1 = create_blocks(Z1)
# blocks_Z2 = create_blocks(Z2)

# Generate candidates
candidates_Z1 = generate_matches(blocks_Z1, Z1)
# candidates_Z2 = generate_matches(blocks_Z2, Z2)

ending_time = time.time()

# Evaluation
print('------------- Evaluation Results --------------')
print(f'Runtime: {ending_time - starting_time} Seconds')
print('------------- First Dataset --------------')
evaluate(candidates_Z1, pd.read_csv('Data/ZY1.csv'))
# print('------------- Second Dataset --------------')
# evaluate(candidates_Z2, pd.read_csv('Data/ZY2.csv'))
