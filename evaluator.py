from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

# BLEU Skoru
def get_bleu_score(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

# ROUGE Skoru
def get_rouge_score(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

# BERT Skoru
def get_bert_score(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="tr")
    return P.mean().item(), R.mean().item(), F1.mean().item()


def evaluate(results_df, task_type):
    """
    Evaluate the results using BLEU, ROUGE, and BERT scores.

    Args:
        results_df (pd.DataFrame): DataFrame containing model results.
        task_type (str): Task type, either "ozettenbasliga" or "basliktanozete".

    Returns:
        pd.DataFrame: Updated DataFrame with evaluation metrics.
    """
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    bert_P_scores = []
    bert_R_scores = []
    bert_F1_scores = []

    for _, row in results_df.iterrows():
        if task_type == "ozettenbasliga":
            reference = row["real_header"]
            candidate = row["generated_header"]
        elif task_type == "basliktanozete":
            reference = row["real_summary"]
            candidate = row["generated_summary"]
        else:
            raise ValueError("Invalid task type")

        # Skip evaluation if generated equals the prompt
        if reference.strip() == candidate.strip():
            bleu_scores.append(0)
            rouge1_scores.append(0)
            rougeL_scores.append(0)
            bert_P_scores.append(0)
            bert_R_scores.append(0)
            bert_F1_scores.append(0)
            continue

        # Calculate scores
        bleu = get_bleu_score(reference, candidate)
        rouge1, rougeL = get_rouge_score(reference, candidate)
        bert_P, bert_R, bert_F1 = get_bert_score(reference, candidate)

        # Append scores
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge1)
        rougeL_scores.append(rougeL)
        bert_P_scores.append(bert_P)
        bert_R_scores.append(bert_R)
        bert_F1_scores.append(bert_F1)

    # Add scores to the DataFrame
    results_df["BLEU"] = bleu_scores
    results_df["ROUGE-1"] = rouge1_scores
    results_df["ROUGE-L"] = rougeL_scores
    results_df["BERT-P"] = bert_P_scores
    results_df["BERT-R"] = bert_R_scores
    results_df["BERT-F1"] = bert_F1_scores

    return results_df