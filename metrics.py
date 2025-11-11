import numpy as np

def levenshtein_distance(s1: list, s2: list) -> int:
    """Calcula a distância de Levenshtein (edição) entre duas listas de tokens."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, token1 in enumerate(s1):
        current_row = [i + 1]
        for j, token2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (token1 != token2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer_wer(ground_truth_list: list, predicted_list: list):
    """
    Calcula o Character Error Rate (CER) e Word Error Rate (WER).
    
    Args:
        ground_truth_list: Lista de strings de referência.
        predicted_list: Lista de strings preditas.
        
    Returns:
        Um tuple (CER, WER).
    """
    total_cer_dist = 0
    total_wer_dist = 0
    total_chars = 0
    total_words = 0

    for gt_text, pred_text in zip(ground_truth_list, predicted_list):
        # 1. Cálculo do CER (comparação caractere por caractere)
        gt_chars = list(gt_text)
        pred_chars = list(pred_text)
        total_cer_dist += levenshtein_distance(gt_chars, pred_chars)
        total_chars += len(gt_chars)

        # 2. Cálculo do WER (comparação palavra por palavra)
        gt_words = gt_text.split()
        pred_words = pred_text.split()
        total_wer_dist += levenshtein_distance(gt_words, pred_words)
        total_words += len(gt_words)
        
    # Evita divisão por zero
    cer = (total_cer_dist / total_chars) * 100 if total_chars > 0 else 0
    wer = (total_wer_dist / total_words) * 100 if total_words > 0 else 0

    return cer, wer
