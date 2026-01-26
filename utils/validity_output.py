# check language is japanese
import langid


# TODO: check for output format 

sample_outdict = {'qid_1': {'明日の天気はどうなる？': '明日の天気は晴れです。'}, 
                  'qid_2': {'旅行に行くなら海と山のどちらを選ぶ？': 'どちらもいいですね。'}, 
                  'qid_3': {'運命は自分で変えられるもの？': 'I dont know.'},
                  'qid_4': {'明日の天気はどうなる？': 'どうでしょう。明日の天気は晴れかもです。'}
                 }

def verify_output(output_dict: dict):
    # output_dict: {q_id: {question: output}}
    verified_output = {}
    
    lang_jp = False
    one_sent = False
    
    for qId, qa in output_dict.items():
        for question, answer in qa.items():
            # verify lanaguage
            lang, confidence = langid.classify(answer)
            if lang == 'ja': lang_jp = True
            else: 
                lang_jp = False
                print(f"NOT JAPANESE -> {lang} ==> {qId}: {question} -- {answer}")
            
            # verify output length
            if '。' not in answer[:-1]:
                one_sent = True
            else: one_sent = False
            
            # TODO: verify if ouput is grammatical (Needed??)
            
            # All verified
            if (lang_jp and one_sent) == True:
                verified_output[qId] = {question: answer}
            
    return verified_output 


# print(verify_output(sample_outdict))

# NOT JAPANESE -> en ==> qid_3: 運命は自分で変えられるもの？ -- I dont know.
# {'qid_1': {'明日の天気はどうなる？': '明日の天気は晴れです。'}, 'qid_2': {'旅行に行くなら海と山のどちらを選ぶ？': 'どちらもいいですね。'}}