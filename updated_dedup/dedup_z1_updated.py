import pandas as pd
import hashlib
from tqdm import tqdm

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

df = pd.read_csv("Data/Z1.csv")

duplicate_id_pairs_z1 = find_and_pair_duplicates(df, 'title')

print("Pairs of duplicate record z1 IDs:")
print("count, lid, rid")
for pair in enumerate(duplicate_id_pairs_z1):
    print(pair)
