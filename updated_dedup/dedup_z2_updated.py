import pandas as pd
import hashlib
from tqdm import tqdm

def hash_record(record):
    """Create a unique hash for a record by concatenating multiple fields and using SHA-256."""
    # Concatenate the fields of interest to form a single string
    concatenated_record = f"{record['name']}|{record['price']}|{record['brand']}|{record['description']}|{record['category']}"
    record_hash = hashlib.sha256(concatenated_record.encode()).hexdigest()
    return record_hash

def find_and_pair_duplicates(df):
    """Identify duplicates in the DataFrame based on multiple columns and return pairs of their IDs."""
    hash_dict = {}
    duplicate_pairs = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing records"):
        record_hash = hash_record(row)

        if record_hash in hash_dict:
            original_id = hash_dict[record_hash]
            duplicate_pairs.append((original_id, row['id']))
        else:
            hash_dict[record_hash] = row['id']

    return duplicate_pairs

df = pd.read_csv("Data/Z2.csv")

duplicate_id_pairs = find_and_pair_duplicates(df)

print("Pairs of duplicate record z1 IDs:")
print("count, lid, rid")
for pair in enumerate(duplicate_id_pairs):
    print(pair)
