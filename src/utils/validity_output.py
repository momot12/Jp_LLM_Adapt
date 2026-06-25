"""
    This file has functions to check the language validity.
    It is incorporated into the run_model() loop. If lang!=ja, the model generates again until lang==ja.
"""
import langid


def check_language(out_sent:str):
    lang, confidence = langid.classify(out_sent)
    if lang == 'ja': 
        lang_jp = True
    else: 
        lang_jp = False
        # print(f"NOT JAPANESE -> {lang}")
    return lang_jp, lang
    