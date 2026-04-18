import os 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sudachipy import dictionary
from src.utils.helper_functions import get_avg_min_max, _get_sent_first_sent


def calc_bleu_score(reference_str, candidate_str):
    """
        Takes in two strings (here: input question and generated output) and calculates BLEU score.
        
        Args:
            reference_str: input
            candidate_str: output   
    """

    # init tokenizer
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer_obj.SplitMode.C
    
    # tokenize into lists of words
    # reference needs to be in list of list
    reference = [[m.surface() for m in tokenizer_obj.tokenize(reference_str, mode)]]
    candidate = [m.surface() for m in tokenizer_obj.tokenize(candidate_str, mode)]
    
    # smoothing function if no overlap n-gram, so total score != 0
    smooth = SmoothingFunction().method1
    
    # BLEU score with brevity penality and smoothing
    score = sentence_bleu(reference, candidate, smoothing_function=smooth)
    
    return score


def write_bleu_sent(data, threshold:float, model_topic_run:str, output_path, mode:int=2):
    """
        Calcualtes BLEU score of data.
        Args:
            data: list of dicts with {question: xx, output: xx}
            mode: 
                1=whole output
                2=first sentence (default)
                3=second sentence
        
    """
    # titles for file
    titles = {1: "Full_Output", 2: "First_Sentence", 3: "Second_Sentence"}
    main_title = titles.get(mode, "Unknown Mode")
    
    info_results = {}
    bleu_counts = {"above_threshold": 0, "exact_repetition": 0}
    
    results, scores = [], []
    for pair in data:
        text = pair.get("only_out", "")
        q_id = pair.get("q_id", "")
        # without punctutation -> only interested in text
        sentences, first_sent = _get_sent_first_sent(text)
        
        cand = ""
        if mode == 1:
            cand = text
        elif mode == 2 and len(sentences) >= 1:
            cand = first_sent
        elif mode == 3 and len(sentences) >= 2:
            cand = sentences[1]
        else:
            cand = first_sent  #fallback to first sent

        # Format the entry
        if cand:
            # remove punctiation 
            cand = cand.strip(r'[。！!？?]')
            # remove ? from question  
            ref = pair["question"].strip('？')
            score = calc_bleu_score(ref, cand)
            entry = f"\nReference: {ref}\nCandidate: {cand}\nQ_ID: {q_id} => Score: {score:.4f}\n"
            results.append(entry)
            scores.append(score)
            
            # count above threshold
            if score > threshold:
                bleu_counts["above_threshold"] += 1
            if score == 1.0:
                bleu_counts["exact_repetition"] += 1
            
            # add collective info dict
            info_results.setdefault("per_q", {})[q_id] = score
    
    avg, v_min, v_max, v_std = get_avg_min_max(scores)
    # Write everything to the file
    outfile = os.path.join(output_path, f"bleu_{model_topic_run}_{main_title.lower()}.txt" )
    # os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(f"Overall: {main_title}\n\n")
        f.write(f"Avg: {avg}\nMin: {v_min}\nMax: {v_max}\nStd: {v_std}\n\n")
        f.write("\n".join(results))
    
    print(f"BLEU scores saved in: {outfile}")
    
    # add collective info dict
    info_results.update({
                        "all_scores": scores, 
                        "avg": avg,
                        "min": v_min,
                        "max": v_max,
                        "std": float(v_std),
                        "bleu_counts": bleu_counts
                        })
    
    return info_results
