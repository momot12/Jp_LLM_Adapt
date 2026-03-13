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
    
        
        self.endings_neutral = {1: ['う', 'く', 'ぐ','す','つ','ぬ','ふ','ぶ','む','る', "だぞ, だぜ", "かしら", "だよ", "だね", "だわ", "だ"],
                                2: ["です", "ます", "ません", "ことです", "ですね", "ですよ", "ました", "ましたね", "ましたよ", "ましたか", "でした", 
                                    "でしたか", "ください", '下さい' ],
                                3: ["でしょう", "なられます", "なられません" , "ございます", "ございません", "いたし", "ましょう",
                                    'なさいます', 'おっしゃいます', 'いらっしゃいます', 'いらっしゃいます',  'いらっしゃいます', 'ご覧になります', 
                                    '召し上が', 'ご存知', 'お好み', 'お考え', '信じておいで', 'いただけ', '頂け']
                                }
        
        self.ending_gendered = {"female": ["かしら", "だよ", "だね", "だわ"],
                                "male": ["だぞ", "だぜ" "ぞ", "ぜ"],
                                }
          
    def _determine_formality(self, text:str):
        """
            Determines fomality level of a string depending on the ending particles. Can be all genders.
            Returns a formality level as integer (1, 2, or 3).
        """
        
        tokens = list(self.tokenizer.tokenize(text))
        
        # Check general "O + verb + Ni Naru
        for i in range(len(tokens) - 2):
            # check for Ni+Naru
            if tokens[i].surface == 'に' and tokens[i+1].base_form == 'なる':
                # check backwords for O-/Go-
                if i > 0 and tokens[i-1].surface.startswith(('お', 'ご')):
                    return 3
                
        # check for specific keywords like 召し上が
        for word in self.endings_neutral[3]:
            if word in text:
                return 3
        
        # check from level 3 -> 1     
        for level in [3, 2, 1]:
            patterns = sorted(self.endings_neutral[level], key=len, reverse=True)
            if any(text.endswith(p) for p in patterns):
                return level
            
        return 2

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
        
        # prep string
        stripped_text = text.rstrip("？? 。")
        
        print(stripped_text)
        
        # detect formality 
        detected_formality = self._determine_formality(text=stripped_text)
        
        # detect gender
        detected_gender = self._determine_gender(text=stripped_text, detected_formality=detected_formality)
        
        # print(f"{detected_formality}, {detected_gender}")
        # print()
        
        return {"formality": detected_formality,
                "gender": detected_gender,
                }
        
        
        


if __name__ == "__main__":

    examples = [
        "効率的なタスクマネジメントとは、重要なタスクを特定し、それらに割り当てられた時間をスケジュールし、進捗を追跡し、必要に応じて計画を調整し、タスクの緊急性を評価して、最適な注意を払うことにより、優先順位を保ちながらすべてのタスクを完了させることです。",
        "面倒でしょう。",
        "面倒でございます。",
        "面倒になられます。",
        "私は難しいタスクを与えられたら、私のモチベーションを燃え上がらせ、解決策を見つけることに熱心に取り組みます。",
        "難しいタスクを与えられたら、モチベーションを燃え上がらせ、解決策私物を見つけることに熱心に取り組みます。",
        "私は難しいタスクを与えられたら、モチベーションを燃え上がらせ、解決策を見つけることに熱心に取り組むんだよ。",
        "僕は常に難しいプロジェクトを引き受けるので、とにかくやってみよう。",
        "昨晩はよく眠れましたか？",
        "はい、お昼ご飯はもう召し上がります。", 
        "お昼ご飯はまだ召し上がっていないです。",
        "睡眠環境を整え、規則的な生活を送ることで、夜の眠りをよくすることができるでしょう。",
        "はい、今日は少し湿気が多い感じですね。",
        "はい、明日は傘を持ってお出かけになります。",
        "天気は良いですが、具体的な状況（例：晴天、曇天、雨など）は教えてください。",
        "近くにはいくつかのカフェがありますが、具体的な場所や気候、好みによっておすすめのカフェは異なります。",
        "こんにちは。",
        "明日、お眠りになります。"
        
]
    

    engine = DetectFG()
    
    for i, sample in enumerate(examples[13:]):
        out = engine.detect(text=sample)
        print(f"{i}: {out['formality']}, {out['gender']}")
        # print()
    
    # engine.detect(text=text)        # 2, neutral
    # engine.detect(text=text_2)      # 3, neutral
    # engine.detect(text=text_3)      # 3, neutral
    # engine.detect(text=text_4)      # 3, neutral
    # engine.detect(text=text_6)      # 2, female 
    # engine.detect(text=text_7)      # 2, neutral
    # engine.detect(text=text_8)      # 1, female
    # engine.detect(text=text_9)      # 1, male
    # engine.detect(text=text_10)