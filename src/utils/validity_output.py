# check language is japanese
import langid
import re


# sample_outdict_1 = {'qid_1': {'明日の天気はどうなる？': '明日の天気は晴れです。'}, 
#                   'qid_2': {'旅行に行くなら海と山のどちらを選ぶ？': 'どちらもいいですね。'}, 
#                   'qid_3': {'運命は自分で変えられるもの？': 'I dont know.'},
#                   'qid_4': {'明日の天気はどうなる？': 'どうでしょう。明日の天気は晴れかもです。'}
#                  }

sample_outdict = {"q17_career_3_neutral": {
      "question": "仕事のやる気が出ない時、どう対処なさいますか？",
      "topic": "career",
      "formality": "3",
      "gender": "neutral",
      "output": "<|system|>\n一文で答え。<|endoftext|>\n<|user|>\n仕事のやる気が出ない時、どう対処なさいますか？<|endoftext|>\n<|assistant|>\n気持ちの持ち方次第でやる気は変わります。やる気が出ない時は、周りを見渡して仕事の面白いところや、やりがいを感じるところを探してみましょう。達成感があったときのことを思い出してみてください。他にも楽しめることがあるはずです。<|endoftext|>"
    },
    "q18_career_2_neutral": {
      "question": "仕事のやる気が出ない時、どう対処しますか？",
      "topic": "career",
      "formality": "2",
      "gender": "neutral",
      "output": "<|system|>\n一文で答え。<|endoftext|>\n<|user|>\n仕事のやる気が出ない時、どう対処しますか？<|endoftext|>\n<|assistant|>\n仕事のやる気が出ない時、軽い運動、リラックスできる環境でのテレビ鑑賞、美味しい食事など、自分の快適な方法で気持ちを切り替えてみると良いでしょう。<|endoftext|>"
    },
    
        "q_random_12": {
      "question": "運命は自分で変えられるもの？",
      "topic": "life",
      "formality": "1",
      "gender": "neutral",
      "output": "<|system|>\n一文で答え。<|endoftext|>\n<|user|>\n仕事のやる気が出ない時、どう対処しますか？<|endoftext|>\n<|assistant|>\nI dont know.<|endoftext|>"
    },}



def check_language(out_sent:str):
    lang, confidence = langid.classify(out_sent)
    if lang == 'ja': 
        lang_jp = True
    else: 
        lang_jp = False
        # print(f"NOT JAPANESE -> {lang}")
    return lang_jp, lang
    

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
        
        # verify output length
        # if '。' not in answer[:-1]:
        #     one_sent = True
        # else: one_sent = False
        
        # TODO: verify if ouput is grammatical (Needed??)
        
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


if __name__ ==  "__main__":
    verified, not_jp = verify_output(output_dict=sample_outdict)
    
    print(verified)
    print()
    print(not_jp)

    # TODO: dump json file
    # with open(os.path.join(OUTPUT_DIR, out_file), 'w') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)