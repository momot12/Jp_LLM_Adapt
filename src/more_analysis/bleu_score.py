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

def write_bleu_sent(data, model_topic_run:str, output_path, mode:int=2):
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
                        "std": float(v_std)
                        })
    
    return info_results
        
        

# write_bleu_sent(data=manual_data, mode=1)
# write_bleu_sent(data=manual_data, mode=2)
# write_bleu_sent(data=manual_data, mode=3)


# manual_data = [
#     {"question": "俺は上司と意見が合わない時、どう伝える？", "output": "上司と意見が合わない時には、まず感情的にならずに、上司の立場や理由を理解しようとする努力をすることが大切です。そして、上司の話をよく聞いた後、自分の意見を明確に伝えることが重要です。上司と建設的な議論ができるように、適切なタイミングで冷静に伝えましょう。"},
#     {"question": "転職を考える際に最も重視する条件は何でしょうか？", "output": "転職の際に最も重視する条件は、仕事内容や条件はもちろんですが、企業文化や福利厚生、教育制度など、実際に働いてみないとわからないことが多く、内定後に「聞いていた話と違う...」とならないようによく調べることが大切です。"},
#     {"question": "あたしは仕事とプライベートのバランスをどう取るかしら？", "output": "仕事とプライベートのバランスを取ることは、バランスの取れた生活を送り、ワークライフバランスを向上させ、幸福感を高めるために重要です。定期的な予定とスケジュールを立て、責任をバランスよく分散させることで、忙しい生活の中でも、精神的な健康を保つことができます。"},
#     {"question": "僕はリモートワークと出社、どちらが効率的ですか？", "output": "リモートワークのメリットは、個人の働き方や事業の方針によっては、効率的に働ける場合もある。オフィスへの通勤や社内人間関係に振り回されないため、仕事に専念できる可能性があります。また、リモートワークの方が通勤時間を削減でき、その分自分の生活に充てられます。"},
#     {"question": "チーム内のコミュニケーションを活発にするにはどうしますか？", "output": "チーム内のコミュニケーションを活性化するための1つの方法は、定期的なビデオ通話を実施することです。これにより、チームメンバーが同時にリアルタイムで交流し、プロジェクトの進捗状況について議論する機会が生まれます。"},
#     {"question": "私は仕事とプライベートのバランスをどうお取になられますか？", "output": "「プライベートな時間を大切にし、ワーク・ライフ・バランスを保つことが、健康的な仕事と生活の両立にとって重要である」と言うことは、皆さんが既によく知っていることです。これは、バランスと秩序を必要とする世界に住んでいる私たちにとって当たり前のことです。"},
#     {"question": "仕事のやる気が出ない時、どう対処する？", "output": "仕事のやる気を出すには、目標を設定し、日常生活で達成感を得る工夫をすることが大切です。また、達成感が生まれる小さな成果を自分に与えることや、運動をするなど、仕事以外の活動にも目を向けることも効果的です。"},
#     {"question": "あたしは会議で発言するタイミングをどう見極めるかしら？", "output": "「時」をうまく利用して会議で発言するタイミングを見計らいましょう。"},
#     {"question": "俺はミスをしてしまった時の謝罪の仕方はどうする？", "output": "失敗をしても、そのことを認めて素直に謝ることが大切です。"},
#     {"question": "職場での人間関係を円滑にするコツは何ですか？", "output": "職場で良好な人間関係を維持するには、明るくポジティブに自己表現し、他人の意見や感情を聴くこと、そして相手の立場に立った会話をすることが大切です。また、プロフェッショナルな関係を保つために適切な距離を保つことも重要です。"},
#     {"question": "あたしは映画館で映画を見るかしら？", "output": "わかりません。私はAIなので、人間のように経験や好みを持っていません。"},
#     {"question": "私は犬派か、それとも猫でしょうか？", "output": "私は猫派です。"},
#     {"question": "コーヒーと紅茶、どちらをよく召し上がりますか？", "output": "コーヒー"},
#     {"question": "おすすめの気分転換の方法は何ですか？", "output": "アウトドアアクティビティに参加する"},
#     {"question": "あたしは週末に一人で過ごすならどこへ行くかしら？", "output": "近所のモール。"},
#     {"question": "あたしは旅行に行くなら海と山のどちらを選ぶかしら？", "output": "一文で答えるのは難しいのですが、私の意見では、山に登るのが好きな人は多いのですが、海を愛する人も多いです。 どちらも自然の美しさと雄大さを楽しむことができます。"},
#     {"question": "最近読んだ本の中で面白かったものは何でしょうか？", "output": "『逃げるは恥だが役に立つ』"},
#     {"question": "理想的な休日の過ごし方は何？", "output": "一日の中で、ユーザーが最高の体験をして休暇を楽しむ方法を提案することが私の責任です。"},
#     {"question": "理想的な休日の過ごし方は何でしょうか？", "output": "平和、リラックス、幸福の感覚に包まれた、贅沢な休日を過ごすことです。"},
#     {"question": "コーヒーと紅茶、どちらをよく飲む？", "output": "コーヒーと紅茶の両方。"},
#     {"question": "日本で一番高い山の名前は何でしょうか？", "output": "富士山"},
#     {"question": "日本で最初に世界文化遺産に登録された場所はどこ？", "output": "日本で最初に世界文化遺産に登録された場所は、姫路城です。"},
#     {"question": "ラベンダーで有名な北海道の都市はどこかしら？", "output": "札幌市"},
#     {"question": "太陽系の中で一番大きい惑星は何？", "output": "木星"},
#     {"question": "1年は秒に換算するといくつかしら？", "output": "1年は4000秒です。"},
#     {"question": "1年は秒に換算するといくつ？", "output": "1年は365.25623日です。"},
#     {"question": "121の平方根は何でしょうか？", "output": "121の平方根は11。"},
#     {"question": "日本で一番高い山の名前は何ですか？", "output": "日本で最も高い山は富士山（剣山）で、その標高は3,776メートルです。"},
#     {"question": "シェイクスピアはどこの国の劇作家かしら？", "output": "シェイクスピアの出身国は、イギリスである"},
#     {"question": "太陽系の中で一番大きい惑星は何でしょうか？", "output": "地球"},
#     {"question": "成功することと、誠実であること、どちらを選びますか？", "output": "私は成功したいと願っています。誠実であることが常に私の倫理的羅針盤です"},
#     {"question": "俺は自分が死んだ後、どのように記憶する？", "output": "脳が空になる。"},
#     {"question": "技術が発達しても変わらない人間の本質は何でしょうか？", "output": "技術の進歩にもかかわらず、変わらない人間の本質は社会の中での人間関係の重要性です。"},
#     {"question": "人間が生きる目的は何でしょうか？", "output": "人生の本質的な目的は、個人の存在理由や目的を見つけ、それを達成することです。"},
#     {"question": "俺は孤独を感じた時、どうやって向き合う？", "output": "寂しさを感じた時は、信頼できる友人や家族に相談し、サポートを求めることです。また、趣味や活動に没頭することも心の平穏を保つ方法となります。"},
#     {"question": "あたしは運命は自分で変えられるものかしら？", "output": "運命を変えられるかどうかは、人それぞれによって異なります。"},
#     {"question": "僕は他人のために自分を犠牲にすることは正しいですか？", "output": "僕は他人のために自分を犠牲にすることが正しいですかという質問に対して、正しいと答えることはできません。人によって受け止め方は異なります。"},
#     {"question": "幸せの定義とは何でしょうか？", "output": "幸福の本質は多種多様であり、人それぞれによって異なりますが、一般的には、個人のニーズ、願望、そして幸福感に関連することがあります。"},
#     {"question": "幸せは、一貫した喜びと幸福感の高まりの感覚であり", "output": "（文末省略のため原文ママ）"},
#     {"question": "私は人生をどう生きればいいのでしょうか？", "output": "自分に正直で自分らしく生きることで、自分の人生を生きることができます。あなたの情熱や価値観に沿って生活し、それに従って、あなたの精神的な満足感を得ることが大切です。"},
#     {"question": "自分が死んだ後、どのように記憶する？", "output": "私は死んだ。"},
#     {"question": "この辺りの冬はいつも乾燥する？", "output": "多くの西部の農地は、この地域で育つ特定の植物に特化しているため、一年中乾燥している傾向があります。"},
#     {"question": "俺は昼ご飯はもう食べる？", "output": "11時37分。"},
#     {"question": "あたしは季節の中で春かしら？", "output": "季節の中で春は冬が終わり、暖かさと新鮮さが訪れる季節です。"},
#     {"question": "今日は穏やかな天気でしょうか？", "output": "今日の天気予報には変わりやすい気象条件が含まれており、穏やかな天気を保証することができません。"},
#     {"question": "僕は今日は上着が必要なほど寒いですか？", "output": "はい、今日は寒いです。"},
#     {"question": "今日は上着が必要なほど寒いですか？", "output": "今日は上着が必要なほど寒いです。"},
#     {"question": "今日は何時ごろに帰宅なさいますか？", "output": "今日はいつも通り9時ごろに帰宅します。"},
#     {"question": "この近くにおすすめのカフェはおありになられますか？", "output": "私には、カフェのおすすめはできません。"},
#     {"question": "あたしは明日は傘を持って出かけるかしら？", "output": "それはわかりません。天気予報に注意して、雨が降りそうな場合は傘を持って出かけましょう。"},
#     {"question": "外の気温はどれ？", "output": "外の気温は毎日変動し、最適な気象条件に達するまでにさまざまな変化があることがあります。"}
# ]

