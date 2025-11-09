import pandas as pd
import json
from pandas import json_normalize
from rapidfuzz import fuzz
from datetime import datetime
from Levenshtein import distance as levenshtein_distance

def blocking(df1: pd.DataFrame, df2: pd.DataFrame):
    # Create a block key from the first 4 letters of the surname
    df1['given_block'] = df1['name_parts'].apply(lambda x: (x.get('surname','')[:4]).lower() if x else '')
    df2['given_block'] = df2['name_parts'].apply(lambda x: (x.get('surname','')[:4]).lower() if x else '')

    blocks = []
    # Iterate over all unique given_name blocks
    unique_blocks = set(df1['given_block']) | set(df2['given_block'])
    for g_block in unique_blocks:
        ids1 = df1[df1['given_block'] == g_block]['id'].tolist()
        ids2 = df2[df2['given_block'] == g_block]['id'].tolist()
        blocks.append((ids1, ids2))

    return blocks


def fuzzy_name_similarity(name1, name2, weights=None):
    # Base weights
    if weights is None:
        weights = {"given-name": 7, "surname": 1, "maiden-surname": 1, "patronymic": 1}

    # Determine if either name has a given-name
    has_given_name = bool(name1.get("given-name")) or bool(name2.get("given-name"))

    # Adjust weights dynamically
    if not has_given_name:
        # No given-name: increase importance of maiden-surname & patronymic
        weights = {"given-name": 0, "surname": 1, "maiden-surname": 3, "patronymic": 3}
        global_weight = 0.2  # global similarity less important
    else:
        global_weight = 0.7  # normal global weight

    part_weight = 1 - global_weight

    score = 0
    total_weight = 0

    for key, weight in weights.items():
        n1 = name1.get(key, "") or ""
        n2 = name2.get(key, "") or ""
        if not n1 and not n2:
            continue

        # Base fuzzy similarity
        part_score = fuzz.token_set_ratio(n1, n2) / 100

        # Flat penalty if lengths differ
        if len(n1) != len(n2):
            part_score *= 0.8

        # Flat penalty if there is exactly one misspelling
        if levenshtein_distance(n1, n2) == 1:
            part_score *= 0.8

        score += weight * part_score
        total_weight += weight

    # Global (combined) name comparison
    full_name_1 = " ".join(filter(None, [name1.get("given-name", ""), name1.get("surname", "")]))
    full_name_2 = " ".join(filter(None, [name2.get("given-name", ""), name2.get("surname", "")]))
    global_score = fuzz.token_set_ratio(full_name_1, full_name_2) / 100

    # Apply penalties to global score
    if len(full_name_1) != len(full_name_2):
        global_score *= 0.8
    elif levenshtein_distance(full_name_1, full_name_2) == 1:
        global_score *= 0.8

    # Handle case when no parts had weights
    if total_weight == 0:
        return global_score

    # Weighted combination: part-based and global
    final_score = (part_weight * (score / total_weight)) + (global_weight * global_score)
    return final_score


def penalize(row):
    n1, n2 = row['name_parts_1'], row['name_parts_2']
    if not (isinstance(n1, dict) and isinstance(n2, dict)):
        return 0.0

    # Base fuzzy name similarity
    score = fuzzy_name_similarity(n1, n2)

    # --- Gender mismatch penalty ---
    g1, g2 = row.get('gender_1'), row.get('gender_2')
    if pd.notna(g1) and pd.notna(g2) and str(g1).strip() != "" and str(g2).strip() != "":
        if str(g1).strip().lower() != str(g2).strip().lower():
            score *= 0.2  # Penalize mismatch

    # --- Birth date mismatch penalty ---
    b1, b2 = row.get('birth_date_1'), row.get('birth_date_2')
    if pd.notna(b1) and pd.notna(b2) and str(b1).strip() and str(b2).strip():
        try:
            d1 = datetime.strptime(str(b1).strip(), "%Y-%m-%d")
            d2 = datetime.strptime(str(b2).strip(), "%Y-%m-%d")
            delta_days = abs((d1 - d2).days)
            if delta_days > 30*12:  # more than 12 months apart
                score *= 0.2
        except ValueError:
            # if date parsing fails, ignore
            pass

    return score



def algo(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df1['name_parts'] = df1.name_parts.apply(json.loads)
    df2['name_parts'] = df2.name_parts.apply(json.loads)
    df1['name_parts'] = df1['name_parts'].apply(lambda d: {k: v.lower() if isinstance(v, str) else v for k, v in d.items()})
    df2['name_parts'] = df2['name_parts'].apply(lambda d: {k: v.lower() if isinstance(v, str) else v for k, v in d.items()})

    blocks = blocking(df1=df1, df2=df2)
    potential_matches = []

    for block in blocks:
        if len(block[0]) == 0 or len(block[1]) == 0:
            continue

        df1_block = df1[df1['id'].isin(block[0])].copy()
        df2_block = df2[df2['id'].isin(block[1])].copy()

        df = df1_block[['id', 'title', 'name_parts', 'gender', 'birth_date']].merge(
            df2_block[['id', 'title', 'name_parts', 'gender', 'birth_date']],
            how='cross',
            suffixes=('_1', '_2')
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]

        # Remove self-matches
        if 'id_1' in df.columns and 'id_2' in df.columns:
            df = df[df['id_1'] != df['id_2']].reset_index(drop=True)

        if not df.empty:
            df['similarity'] = df.apply(penalize, axis=1)
            df = df[df['similarity'] >= threshold]
            potential_matches.append(df)

    if potential_matches:
        matches = pd.concat(potential_matches, ignore_index=True)
    else:
        return pd.DataFrame(columns=['id_1', 'title_1', 'name_parts_1', 'id_2', 'title_2', 'name_parts_2', 'similarity'])

    matches = matches.sort_values('similarity', ascending=False)

    return matches[['id_1', 'title_1', 'name_parts_1', 'id_2', 'title_2', 'name_parts_2', 'similarity']]


def evaluate(path_to_ts1, path_to_ts2, path_to_em, threshold):
    df1 = pd.read_csv(path_to_ts1, sep='\t')
    df2 = pd.read_csv(path_to_ts2, sep='\t')
    em = pd.read_csv(path_to_em, sep='\t')

    em[['id_1', 'id_2']] = em[['id_1', 'id_2']].astype(str).apply(lambda x: x.str.strip())
    results = algo(df1=df1, df2=df2, threshold=threshold)
    results[['id_1', 'id_2']] = results[['id_1', 'id_2']].astype(str).apply(lambda x: x.str.strip())

    results['pair'] = results.apply(lambda row: tuple(sorted((row['id_1'], row['id_2']))), axis=1)
    em['pair'] = em.apply(lambda row: tuple(sorted((row['id_1'], row['id_2']))), axis=1)

    results_unique = results.drop_duplicates(subset=['pair'])
    em_unique = em.drop_duplicates(subset=['pair'])

    predicted_pairs = set(results_unique['pair'])
    true_pairs = set(em_unique['pair'])

    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    beta = 5
    f5 = ((1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)) if (precision + recall) > 0 else 0.0

    fp_rows = results_unique[~results_unique['pair'].isin(true_pairs)]
    missing_pairs = true_pairs - predicted_pairs
    fn_rows = em_unique[em_unique['pair'].isin(missing_pairs)]
    
    print("=== Evaluation Results ===")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"F5-score: {f5:.4f}")
    print("\n=== False Positives ===")
    print(fp_rows)
    print("\n=== False Negatives ===")
    print(fn_rows)
    
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "F5": f5,
        "False_Positives": fp_rows
    }


eval_results = evaluate('test_resources/testset13-YadVAshemItaly/yv_italy.tsv',
         'test_resources/testset14-YadVashemItalySample/yv_italy_sampled_records.tsv',
         'test_resources/testset14-YadVashemItalySample/em_sample.tsv',
         threshold=0.9)

