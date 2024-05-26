import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

aliases_pattern = {
    'class': ['classe', 'clase', 'clas ', 'klasse', 'cl '],
    'uhsi': ['uhs1', 'uhs-i', 'ultra high-speed'],
    'type-c': ['typec', 'type c', 'usb-c', 'usbc'],
    'extreme': ['extrem'],
    'att4': ['attach'],
    'adapter': ['adapter', 'adaptateur', 'adaptador', 'adattatore'],
    'memory': ['memoria', 'mémoire', 'memória', 'memoria', 'memory', 'geheugen', 'memoría', 'menor', 'mem', 'memoire', 'memoria'],
    'flash drive': ['usb stick', 'usb flash drive', 'pen drive', 'usb drive', 'flash drive', 'flash disk', 'usb flash', 'usb stick', 'pen drive', 'usb-flash', 'pendrive', 'memory stick'],
    'hard drive': ['harddisk', 'hard disk', 'hdd', 'external drive', 'external hard drive', 'hard disk drive', 'harddrive', 'disk drive'],
    'ssd': ['solid state drive', 'ssd drive', 'solid-state drive', 'ssd disk'],
    'micro sd': ['microsd', 'micro sd card', 'micro sdxc', 'micro sd hc', 'micro sdhc', 'micro sdxc'],
    'sd card': ['sdcard', 'sd card', 'sd memory card', 'sdhc', 'sdxc', 'secure digital card'],
    'usb': ['usb3', 'usb2', 'usb 3.0', 'usb 2.0', 'usb 3', 'usb 2', 'usb-c', 'usbc', 'type-c', 'type c'],
    'portable': ['portátil', 'portatif', 'portatile', 'portabel', 'tragbar'],
    'wireless': ['wireless', 'wireless', 'sin cables', 'sans fil', 'kabellos', 'draadloos', 'senza fili'],
    'charger': ['charging', 'ladegerät', 'chargeur', 'caricatore', 'lader', 'charg'],
    'camera': ['camcorder', 'kamera', 'cámara', 'caméra', 'fotocamera', 'videocamera', 'webcam', 'cam'],
    'lens': ['objective', 'objektiv', 'lente', 'lentille', 'objetivo', 'linsen', 'lins'],
    'screen': ['display', 'monitor', 'scherm', 'écran', 'pantalla', 'bildschirm'],
    'battery': ['akku', 'batteria', 'batería', 'batterie', 'batterij', 'bat'],
    'phone': ['smartphone', 'telefono', 'téléphone', 'telefone', 'handy', 'mobiltelefon', 'handtelefon', 'cell phone'],
    'laptop': ['notebook', 'portátil', 'ordinateur portable', 'laptop', 'tragbarer computer'],
    'tablet': ['tab', 'slate', 'pad', 'pills', 'tablette', 'tableta', 'tablet computer']
}

brand_pattern = r'\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b'

model_patterns = [
    r'\b([\(]*[\w]+[-]*[\d]+[-]*[\w]+[-]*[\d+]*|[\d]+[\w]|[\w][\d]+)',
    r'\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!=sim)|evo|xqd|ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\s?9|ultra)',
    r'(thn-[a-z][\w]+|ljd[\w+][-][\w]+|ljd[sc][\w]+[-][\w]+|lsdmi[\d]+[\w]+|lsd[0-9]{1,3}[gb]+[\w]+|ljds[0-9]{2}[-][\w]+|usm[0-9]{1,3}[\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\w]+|sdsd[a-z]+[-][0-9]+[\w]+[-]*[\w]*|sdcz[\w]+|mk[\d]+|sr-g1[\w]+)',
    r'\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})'
]

color_pattern = r'\b(black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey|silver|gold|beige|ivory|turquoise|violet|navy|teal|maroon|burgundy|magenta|cyan|lime|olive)\b'

# Precompile patterns, enhances perfomance
clean_brand = re.compile(brand_pattern)
clean_models = [re.compile(pattern) for pattern in model_patterns]
clean_color = re.compile(color_pattern)

# Function to apply aliases
def apply_aliases(text, aliases_pattern):
    for key, aliases in aliases_pattern.items():
        for alias in aliases:
            text = re.sub(r'\b' + re.escape(alias) + r'\b', key, text)
    return text

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    text = apply_aliases(text, aliases_pattern)  # Apply aliases
    return text

# Function to generate blocking key name
def generate_blocking_key_name(row: pd.Series):
    keys = []
    
    # Process name field
    if pd.notna(row['name']):
        name_cleaned = clean_text(str(row['name']))
        matches = clean_brand.findall(name_cleaned)
        keys.extend([match.lower() for match in matches])
        for pattern in clean_models:
            matches = pattern.findall(name_cleaned)
            keys.extend([match.lower() for match in matches])
        matches = clean_color.findall(name_cleaned)
        keys.extend([match.lower() for match in matches])
    
    # Process brand field
    if pd.notna(row['brand']):
        brand_cleaned = clean_text(str(row['brand']))
        matches = clean_brand.findall(brand_cleaned)
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
