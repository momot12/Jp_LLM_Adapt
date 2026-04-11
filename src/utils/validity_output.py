"""
    This file has functions to check the language validity.
    It is incorporated into the run_model() loop. If lang!=ja, the model generates again until lang==ja.
    
    TODO (15.03.26): verify_output() might be ok to delete; decide later.
"""

import langid
import re

def check_language(out_sent:str):
    lang, confidence = langid.classify(out_sent)
    if lang == 'ja': 
        lang_jp = True
    else: 
        lang_jp = False
        # print(f"NOT JAPANESE -> {lang}")
    return lang_jp, lang
    

# NOTE: probably not needed anymore (15.03.26)
def verify_output(output_dict: dict):
    # output_dict: {q_id: {question:xx, topic:xx, formality:x, gender:x, output:xxx}}

    verified_output = {}
    not_jp_out = {}
    
    lang_jp = False
    one_sent = False
    
    for qId, qa in output_dict.items():
        match = re.search(r"<\|assistant\|>\n?(.*?)(?=<\|endoftext\|>|$)", 
                    qa["output"], re.DOTALL)
        output_clean = match.group(1).strip() if match else ""
        
        # verify lanaguage
        lang, confidence = langid.classify(output_clean)
        if lang == 'ja': lang_jp = True
        else: 
            lang_jp = False
            print(f"NOT JAPANESE -> {lang} ==> {qId}:{output_clean}")
  
        # All verified
        # if (lang_jp and one_sent) == True:
        if lang_jp:
            verified_output[qId] = {
                    "question": qa["question"],
                    "topic": qa["topic"],
                    "formality": qa["formality"],
                    "gender": qa["gender"],
                    "output": qa["output"]
                    }
        else:
            not_jp_out[qId] = {
                    "question": qa["question"],
                    "topic": qa["topic"],
                    "formality": qa["formality"],
                    "gender": qa["gender"],
                    "output": qa["output"]
                    }
            
    return verified_output, not_jp_out