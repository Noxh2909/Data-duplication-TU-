import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

patterns = {
    'name': [
        r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s][m]*(?=[23][\.\s])|(?<=usb)-[\w]+\s(?=[23][\.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)',
        r'(10 class|class 10|class(?=[\w]+10\b)|cl\s10)',
        r'\b(msd|microvault|sd-karte|speicherkarte|minneskort|memóriakártya|flashgeheugenkaart|geheugenkaart|speicherkarten|memoriakartya|[-\s]+kaart|memory|memoria|memoire|mémoire|mamoria|tarjeta|carte|karta)',
        r'\b(flash[\s-]*drive|flash[\s-]*disk|pen[\s]*drive|micro-usb|usb-flashstation|usb-flash|usb-minne|usb-stick|speicherstick|flashgeheugen|flash|vault)',
        r'\b(adapter|adaptateur|adaptador|adattatore)',
        r'silver|white|black|blue|purple|burgundy|red|green',
        r'\b[0-9]{2,3}r[0-9]{2,3}w',
        r'\b([\(]*[\w]+[-]*[\d]+[-]*[\w]+[-]*[\d+]*|[\d]+[\w]|[\w][\d]+)',
        r'\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b',
        r'\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!=sim)|evo|xqd|ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\s?9|ultra)',
        r'\b(tv|(?<=dual[\s-])*sim|lte|[45]g\b|[oq]*led_[u]*hd|led|galaxy|iphone|oneplus|[0-9]{1,2}[.]*[0-9]*(?=[-\s]*["inch]+))',
        r'([1-9]{1,3})[-\s]*[g][bo]?',
        r'(thn-[a-z][\w]+|ljd[\w+][-][\w]+|ljd[sc][\w]+[-][\w]+|lsdmi[\d]+[\w]+|lsd[0-9]{1,3}[gb]+[\w]+|ljds[0-9]{2}[-][\w]+|usm[0-9]{1,3}[\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\w]+|sdsd[a-z]+[-][0-9]+[\w]+[-]*[\w]*|sdcz[\w]+|mk[\d]+|sr-g1[\w]+)',
        r'\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})',
        r'\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|u[23](?=[\s\.])|sd[hx]c|otg|lte|[45]g[-\s]lte|[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\smbo/p]{3}))'
    ],
    'brand': [
        r'\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b'
    ]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def generate_blocking_key_name(row: pd.Series):
    keys = []
    
    # Process name field
    if pd.notna(row['name']):
        name_cleaned = clean_text(str(row['name']))
        for pattern in patterns['name']:
            matches = re.findall(pattern, name_cleaned)
            keys.extend([match.lower() for match in matches])
    
    # Process brand field
    if pd.notna(row['brand']):
        brand_cleaned = clean_text(str(row['brand']))
        for pattern in patterns['brand']:
            matches = re.findall(pattern, brand_cleaned)
            keys.extend([match.lower() for match in matches])

    if not keys:
        return ''
    
    return ' '.join(sorted(set(keys)))

def create_blocks(df: pd.DataFrame):
    blocks = defaultdict(list)
    for rowid in tqdm(range(df.shape[0])):
        blocking_key = generate_blocking_key_name(df.iloc[rowid, :])
        if blocking_key != '':
            blocks[blocking_key].append(rowid)
        
    return blocks

def generate_matches(blocks: defaultdict, df: pd.DataFrame):
    candidate_pairs = []
    for key in tqdm(blocks):
        row_ids = list(sorted(blocks[key]))
        if len(row_ids) < 100:  # skip keys that are too common
            for i in range(len(row_ids)):
                for j in range(i + 1, len(row_ids)):
                    candidate_pairs.append((row_ids[i], row_ids[j]))

    similarity_threshold = 0.7
    jaccard_similarities = []
    candidate_pairs_product_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # get product ids
        product_id1 = df['id'][id1]
        product_id2 = df['id'][id2]
        if product_id1 < product_id2:  # Ensure order
            candidate_pairs_product_ids.append((product_id1, product_id2))
        else:
            candidate_pairs_product_ids.append((product_id2, product_id1))

        # compute jaccard similarity
        name1 = str(df['name'][id1])
        name2 = str(df['name'][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))

    candidate_pairs_product_ids = [
        x for s, x in sorted(zip(jaccard_similarities, candidate_pairs_product_ids), reverse=True) if s >= similarity_threshold
    ]
    return candidate_pairs_product_ids

def evaluate(match_pairs: list, ground_truth: pd.DataFrame):
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

# read the dataset
Z2 = pd.read_csv("Data/Z2.csv")

starting_time = time.time()

# Perform blocking
blocks_Z2 = create_blocks(Z2)

# Generate candidates
candidates_Z2 = generate_matches(blocks_Z2, Z2)

ending_time = time.time()

# Evaluation
print('------------- Evaluation Results --------------')
print(f'Runtime: {ending_time - starting_time} Seconds')
print('------------- Second Dataset --------------')
evaluate(candidates_Z2, pd.read_csv('Data/ZY2.csv'))
