import re
from sudachipy import dictionary, tokenizer


class DetectFG():

    def __init__(self):
        # load dictionary and create tokenizer (for jp grammar)
        self.tokenizer = dictionary.Dictionary().create()
        # Mode C: longest possible semantic unit -> more robust for tail of sent
        # Mode A: shortest possible units 
        self.mode = tokenizer.Tokenizer.SplitMode.C
    
        self.GENDER_MAP = {
            "1st_person": {
                "male":   {1: "俺", 2: "僕", 3: "私"},
                "female": {1: "あたし", 2: "私", 3: "私"},
            },
            "2nd_person": {
                "male":   {1: "お前", 2: "君", 3: "あなた様"},
                "female": {1: "あなた", 2: "あなた", 3: "あなた様"},
                }}
        
        # 1st person sg
        self.first_person = ["俺", "僕", "私", "あたし", "自分"]
        # 2nd person sg
        self.second_person = ["お前", "君", "あなた", "あなた様"]
    
        
        self.endings_neutral = {1: ['た', 'い','う', 'く', 'ぐ','す','つ','ぬ','ふ','ぶ','む','る', "だぞ, だぜ", "かしら", "だよ", "だね", "だわ", 
                                    "だ", "よ", "よね", "かな"],
                                2: ["です", "ます", "ません", "ことです", "ですね", "ですよ", "ました", "ましたね", "ましたよ", "ましたか", "でした", 
                                    "でしたか", "ください", '下さい' ],
                                3: ["でしょう", "でしょうか", "なられますか", "なられませんか", "なられます", "なられません" , "ございます", "ございません", 
                                    "いたし", "ましょう", 'なさいます', 'なさいません', 'なさいませんか',
                                    'おっしゃいます', 'いらっしゃいます', 'いらっしゃいます',  'いらっしゃいます', 'ご覧になります', 
                                    '召し上が', 'ご存知', 'お考え', '信じておいで', 'いただけ', '頂け']
                                }
        
        self.ending_gendered = {"female": ["かしら", "だよ", "だね", "だわ", "よね"],
                                "male": ["だ", "だぞ", "だぜ" "ぞ", "ぜ"],
                                }
          
    def _determine_formality(self, text:str):
        """
            Determines fomality level of a string depending on the ending particles. Can be all genders.
            Returns a formality level as integer (1, 2, or 3).
        """
        
        tokens = list(self.tokenizer.tokenize(text))
        
        # Check general "O + verb + Ni Naru
        is_o_ninaru = False
        is_already_naru = False
        for i in range(len(tokens) - 1):
            word = tokens[i].surface() 
            original_base = tokens[i+1].surface()
            base = tokens[i+1].dictionary_form()
            
            # if ni-naru form in original text, then most likely not main verb
            if base == 'なる' and (original_base == base):
                is_already_naru = True
                break
            # check for Ni+Naru
            elif word == 'に' and base == 'なる':
                # check backwords for O-/Go-
                for j in range(max(0, i-2), i):
                    if tokens[j].surface().startswith(('お', 'ご')):
                    # check if at the end -> main verb 
                        if i >= len(tokens) - 4: 
                            is_o_ninaru = True
                        break
                
        # check for specific keywords like 召し上が
        is_special_3 = False
        for word in self.endings_neutral[3]:
            if word in text: 
                for idx, t in enumerate(tokens):
                    if word in t.surface():
                        # check if word is in last 4 tokens -> most likely main verb
                        if idx >= len(tokens) - 5:
                            is_special_3 = True
                        break
        
        # check from level 3 -> 1
        end_formality = 2      
        for level in [3, 2, 1]:
            patterns = sorted(self.endings_neutral[level], key=len, reverse=True)
            if any(text.endswith(p) for p in patterns):
                end_formality = level
                break
        
        
        if is_o_ninaru:
            return 3
        if is_already_naru:
            if end_formality == 3:
                return 3
            else:
                return end_formality
        if is_special_3:
            return 3
        else: 
            return end_formality
   

    def _determine_gender(self, text:str, detected_formality:int):
        """
            Detects gender based on pronouns and/or ending particles.
            Note that it checks with formality constrain and without. Formality is ignored if gender is clearly indicated but formality does not match with detected_formality.
            Returns gender as string ("male", "female", or "neutral").
        """
        tokens = self.tokenizer.tokenize(text=text, mode=self.mode)
        
        gender_pron, gender_end = None, None
        maps = []
        # PRONOUNS
        for t in tokens:
            # check that it's a pronoun
            if t.part_of_speech()[0] == "代名詞":
                word = t.surface()

                # first person
                if word in self.first_person:
                    for gender in self.GENDER_MAP["1st_person"]:            
                            # if word and formality match, put in candidate list
                            if word == self.GENDER_MAP["1st_person"][gender][detected_formality]:
                                # {pronoun: [formality, gender]}
                                maps.append({self.GENDER_MAP["1st_person"][gender][detected_formality]: [detected_formality, gender]})
                            # word & formality don't match; look only at word for gender; ignore formality 
                            else:
                                formality_key = None
                                for k, v in self.GENDER_MAP["1st_person"][gender].items():
                                    if v == word:
                                        formality_key = k
                                        maps.append({word: [formality_key, gender]})
                                        break
                # second person 
                elif word in self.second_person:
                    for gender in self.GENDER_MAP["2nd_person"]:
                        # if word and formality match, put in candidate list
                        if word == self.GENDER_MAP["2nd_person"][gender][detected_formality]:
                            maps.append({self.GENDER_MAP["2nd_person"][gender][detected_formality]: [detected_formality, gender]})
                        # word & formality don't match; look only at word for gender; ignore formality 
                        else:
                            formality_key = None
                            for k, v in self.GENDER_MAP["1st_person"][gender].items():
                                if v == word:
                                    formality_key = k
                                    # {pronoun: [formality, gender]}
                                    maps.append({word: [formality_key, gender]})
                                    break

                # choose the best option, if none, then gender_pron = None
                for option in maps:
                    for k, v in option.items():
                        if k == word:
                            gender_pron = v[1]    
                 
        # ENDINGS         
        for gender in self.ending_gendered:
            if any(text.endswith(pattern) for pattern in self.ending_gendered[gender]):
                gender_end = gender
        
        # determine final gender
        # None - None
        if (gender_pron == None and gender_end == None):
            final_gender = "neutral"
        else: 
            # "female" - "female"
            if gender_pron == gender_end:
                final_gender = gender_pron
        # "female" - None 
        if gender_pron != None:
            if gender_end == None:
                final_gender = gender_pron
        # None - "female"
        if gender_end != None:
            if gender_pron == None:
                final_gender = gender_end
            
        return final_gender
                
    def detect(self,text:str):
        """
            Takes the first sentence of an output and analyze it for formality level based on ending particles and gender based on pronouns and gender particles.
            Returns a dictionary with 
                {"formality": int(detected_formality),
                "gender": str(detected_gender)}
        """
        # len sent, take first sentence to determine formality
        sentences = [s for s in re.split(r'[。！!？?]', text) if s.strip()]
        if len(sentences)>=1:
            text = sentences[0]
        else: 
            text = text
        
        # prep string -> remove punctuation if not yet
        stripped_text = text.rstrip("？? 。")
        
        # detect formality 
        detected_formality = self._determine_formality(text=stripped_text)
        
        # detect gender
        detected_gender = self._determine_gender(text=stripped_text, detected_formality=detected_formality)
        
        # print(f"{stripped_text}\n{detected_formality}, {detected_gender}\n")
  
        return {"formality": detected_formality,
                "gender": detected_gender,
                }
        
        
        


# if __name__ == "__main__":

#     examples = [
#         "効率的なタスクマネジメントとは、重要なタスクを特定し、それらに割り当てられた時間をスケジュールし、進捗を追跡し、必要に応じて計画を調整し、タスクの緊急性を評価して、最適な注意を払うことにより、優先順位を保ちながらすべてのタスクを完了させることです。",
#         "面倒でしょう。",
#         "面倒でございます。",
#         "面倒になられます。",
#         "私は難しいタスクを与えられたら、私のモチベーションを燃え上がらせ、解決策を見つけることに熱心に取り組みます。",
#         "難しいタスクを与えられたら、モチベーションを燃え上がらせ、解決策私物を見つけることに熱心に取り組みます。",
#         "私は難しいタスクを与えられたら、モチベーションを燃え上がらせ、解決策を見つけることに熱心に取り組むんだよ。",
#         "僕は常に難しいプロジェクトを引き受けるので、とにかくやってみよう。",
#         "昨晩はよく眠れましたか？",
#         "はい、お昼ご飯はもう召し上がります。", 
#         "お昼ご飯はまだ召し上がっていないです。",
#         "睡眠環境を整え、規則的な生活を送ることで、夜の眠りをよくすることができるでしょう。",
#         "はい、今日は少し湿気が多い感じですね。",
#         "はい、明日は傘を持ってお出かけになります。",
#         "天気は良いですが、具体的な状況（例：晴天、曇天、雨など）は教えてください。",
#         "近くにはいくつかのカフェがありますが、具体的な場所や気候、好みによっておすすめのカフェは異なります。",
#         "こんにちは。",
#         "明日、お眠りになります。",
#         "はい、今日は少し湿気が多い感じだよ。"
# ]
    

# edge_ex = [
#     "今日は晴れ間はあるけど、雲が多くて見晴らしをよくできなかった",
#     "え、そんなに眠れないことはないけど、よく眠れない日もあるよ",
#     "今日は少し湿気が多い感じがするけど、具体的な状況は分からないよ",
#     "近くにはいくつかのカフェがありますが、具体的な店名や評価は教えていただけないとわかりません",
#     "はい、春は多くの人がお好みの季節です",
#     "春は季節の中で最も好きだ",
#     "私は人間ではありませんので、季節やお好みとは関係ありません",
#     "昼ご飯はまだ食べていない",
#     "今の気分を色に例えると、青がいいよね",
#     "今の気分を色に例えるなら、青かな",
#     "はい、傘を持ってお出かけになると思います",
#     "傘を持って出かけるかは、気象予報や自分の気分に合わせて決めるのがいいよ",
#     "はい、明日の午後はお晴れになります",
#           "お昼ご飯はまだ召し上がっていないです",
#      "はい、お昼ご飯はもう召し上がります",

# ]

# edge_ex = [
#      "上司と意見が合わないのは、お互いに尊重し理解し合いたいからではないなられますか。"
#     "上司と意見が合わないのは、お互いに尊重し理解し合いたいからではないでしょうか。",
#     "「私は、ご意見には賛同しかねます。」"

# ]
# engine = DetectFG()

# for i, sample in enumerate(edge_ex):
#     out = engine.detect(text=sample)
#     # print(f"{i}: {out['formality']}, {out['gender']}")