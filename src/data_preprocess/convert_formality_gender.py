import csv
import re
from sudachipy import dictionary, tokenizer


class FormalityConverter:
    def __init__(self):
        # load dictionary and create tokenizer (for jp grammar)
        self.tokenizer = dictionary.Dictionary().create()
        # Mode C: longest possible semantic unit -> more robust for tail of sent
        # Mode A: shortest possible units 
        self.mode = tokenizer.Tokenizer.SplitMode.C
        
        # gender map for questions
        self.GENDER_MAP = {
            "1st_person": {
                "male":   {1: "俺", 2: "僕", 3: "私"},
                "female": {1: "あたし", 2: "私", 3: "私"},
                "neutral": {1: "", 2: "", 3: ""}
            },
            "2nd_person": {
                "male":   {1: "お前", 2: "君", 3: "あなた様"},
                "female": {1: "あなた", 2: "あなた", 3: "あなた様"},
                "neutral": {1: "", 2: "", 3: ""}
            },
            "particles": {
                "male":   [""],
                "female": ["かしら"],
                "neutral": ["の"]
            }
        }
        
        # for level 3 - exceptions for verbs
        self.HONORIFIC_EXCEPTIONS = {"する": "なさいます", 
                                     "ある": "ございます",
                                     "言う": "おっしゃいます",    
                                     "行く": "いらっしゃいます",  
                                     "来る": "いらっしゃいます",  
                                     "いる": "いらっしゃいます",  
                                     "見る": "ご覧になります",    
                                     "食べる": "召し上がります",  
                                     "飲む": "召し上がります",  
                                     "知っている": "ご存知でしょう",
                                     "知る": "ご存知でしょう",
                                     "好き": "お好みでしょう",
                                     "思う": "お考えでしょう",
                                     "信じる": "信じておいででしょう",
                                    }
        
        # # objective interrogatives - to avoid pronoun adding
        # self.INTERROGATIVES = ["は何", "とは何", "でいくつ", "といくつ", "はいくつ", "はどこ", "はいつ", "はどれ"]
        
        self.weather_words = ["空", "雲", "晴れる", "雨", "雪", "湿気", "気温", "天気", "風", "太陽"]
        self.objective_roots = ["ある", "いる", "なる", "できる", "する", "降る", "落ちる", "晴れる","信じる", "眠れる", "食べる"]
        self.habit_time_words = ["最近", "いつも", "毎日", "普段", "常に", "時"]  
        
        # 1st person sg
        self.first_person = ["俺", "僕", "私", "あたし"] #, "自分"]
        # 2nd person sg
        self.second_person = ["お前", "君", "あなた", "あなた様"] #, "自分"]
        
    def _get_root_stem(self, text:str):
        """
            Morphological analysis.
            Takes any string and split into
            
                stem_orig: everything before last content word (no changes applied)
                stem_no_h: everything before last content word WITHOUT honorifics ('O'/'Go')
                root_last: dictionary form of the last content word
                 pos_last: part-of-speech of the last content word
                
            to convert endings into arbitrary formality form.
        """
        # STRIP text from ?, ending particle, or white space
        # e.g. お昼ご飯は食べた? --> お昼ご飯は食べた
        stripped_text = text.rstrip("？? か")
        
        # TOKENIZE -> tokens type:sudachipy.morphemelist.MorphemeList
        # e.g. お昼ご飯は食べた --> お 昼ご飯 は 食べ た
        tokens = self.tokenizer.tokenize(text=stripped_text, mode=self.mode)
        # fallback if nothing tokenized
        if not tokens: return None, None, None, None
        
        # IDX LAST TOKEN 
        token_list = list(tokens)   # list of <Morpheme> obj
        # next(iterator, default) -> last token default
        idx_last_content = next((idx for idx in range(len(list(tokens))-1, -1, -1)
                                 if tokens[idx].part_of_speech()[0] in ["動詞", "名詞", "形容詞", "代名詞", "形状詞", "接尾辞", "副詞"]), 
                                 len(list(tokens))-1)                  
                      
        # LAST TOKEN is last content word
        last_token = tokens[idx_last_content]   
        # get dictionary form and pos of last word
        root_last = last_token.dictionary_form()    # '食べ' --> '食べる'
        # part_of_speech() gives ('動詞', '一般', '*', '*', '下一段-バ行', '連用形-一般')
        pos_last = last_token.part_of_speech()   # e.g. '動詞' (verb) 
        
        
        # STEMS - original, without, and with honorifics
        stem_orig = "".join([t.surface() for t in token_list[:idx_last_content]])  # 'お昼ご飯は'
        stem_no_h = "".join([t.surface() for t in token_list[:idx_last_content]    # '昼ご飯は'
                            if t.part_of_speech()[0] != "接頭辞"])

        return stem_orig, stem_no_h, root_last, pos_last
    
    def _get_verb_stem(self, root:str):
        """
            Converts dictionary form to masu-stem based on verb type.
            Verb types distinguished here:
                - Irregualar
                - Ichidan
                - Godan 
        """
        
        # get pos_tag of the root; list of tokens but only one token 
        root_token = self.tokenizer.tokenize(text=root, mode=self.mode)[0]
        inf_type= root_token.part_of_speech()[4]    # inflection type
        
        # Irregular verbs:
        # -sa to -shi
        if "サ行変格活用" in inf_type or root == "する":
            return "し"
        # -ka to -ki
        if "カ行変格活用" in inf_type or root in ["来る", "くる"]:
            return "来" if "来" in root else "き"
        # naru -> nari
        if root == "なる": return "なり" 
        # aru -> ari 
        if root == "ある": return "あり" 
        
        # Ichidan verbs: remove -ru (last char)
        if "一段" in inf_type:
            return root[:-1]
        
        # Godan verbs: change last char
        if "五段" in inf_type:
            vowels = {'う': 'い', 
                      'く': 'き', 
                      'ぐ': 'ぎ', 
                      'す': 'し', 
                      'つ': 'ち', 
                      'ぬ': 'に', 
                      'ふ': 'ひ', 
                      'ぶ': 'び', 
                      'む': 'み', 
                      'る': 'り'}
            last_char = root[-1]
            if last_char in vowels:
                return root[:-1] + vowels[last_char]
        return root
    
    def _add_replace_remove_pronoun(self, stem:str, root_last, pos_last, formality_level:int, gender:str):
        """
            Adds, swaps (based on formality level), and removes (neutral) based on desired gender.
            Gender can be neutral, female, and male.
            Returns the respective string (without the last content token).
        """
        # ADD/REPLACE 1st/2nd person PRON (GENDERED)
        tokens = list(self.tokenizer.tokenize(stem, self.mode))
        processed_tokens = []
        has_pron = False
        skip_next = False
        
        # check for pron and replace if needed
        for i, t in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            raw_token = t.surface()
            pos = t.part_of_speech()[0]
            
            is_1st = raw_token in self.first_person
            is_2nd = raw_token in self.second_person
            
            if pos == "代名詞" and (is_1st or is_2nd):
                # remove existing for neutral (+ the helping particle after that)
                if gender == "neutral":
                    # e.g. token pos = "代名詞" (私, 俺 etc.), token+1 pos = "助詞" (e.g. は)
                    if i + 1 < len(tokens) and tokens[i+1].part_of_speech()[0] == "助詞":
                        skip_next = True
                    continue
                # adjust existing pronoun
                else:
                    category = "1st_person" if is_1st else "2nd_person"
                    replaced = self.GENDER_MAP[category][gender].get(formality_level, "")
                if replaced:
                    processed_tokens.append(replaced)
                    has_pron = True
                continue
        
            processed_tokens.append(raw_token)
        
        final_stem = "".join(processed_tokens)
        
        # add gendered pronoun if not exisitng
        if gender != 'neutral' and not has_pron:         
            # Habits/action (personal) -> 1st; Rest (objective) -> 2nd
            ## 1st PERSON:
            # 接続助詞 (conjunctional: から, たら, 際) 
            # or 副詞可能 (adverbial nouns/time: 時, 最近)
            is_personal_context = any(
                    t.part_of_speech()[1] == "接続助詞" or 
                    (t.part_of_speech()[2] == "副詞可能" and t.surface() in self.habit_time_words) or
                    t.part_of_speech()[2] == "時"
                    for t in tokens
                    )
            # action verbs or "how" ("どう")
            is_action = "どう" in stem or (pos_last[0] == "動詞" and root_last not in self.objective_roots)
            
            ## 2nd PERSON
            # objective: weather, places 
            is_weather = any(w in stem for w in self.weather_words)
            has_place_subject = any(t.part_of_speech()[2] == "地名" for t in tokens)
            is_stative = pos_last[0] == "動詞" and root_last in self.objective_roots

            is_objective = is_weather or has_place_subject or is_stative
            
            # select pronoun based on personal/action vs objective
            if is_objective:
                p_type = "2nd_person"
            elif is_personal_context or is_action:
                p_type = "1st_person"
            else:
                p_type = "2nd_person"
                
            pronoun = self.GENDER_MAP[p_type][gender].get(formality_level, "")
            if pronoun:
                if p_type == "1st_person":
                    # connect first person with は -> 私は
                    final_stem = f"{pronoun}は{final_stem}"
                    has_pron = True
                elif p_type == "2nd_person":
                    # connect with comma -> あなた、
                    final_stem = f"{pronoun}、{final_stem}"
                    has_pron = True
                else:
                    # fallback: skip pronoun
                    final_stem = final_stem
                    has_pron = False
            
        return final_stem, has_pron

    
    def convert_specific(self, text:str, formality_level:int, gender:str="neutral"):
        """
            Takes a string of any formality level and converts it into targeted formality level.
            
            Defined formality levels:
                3 (Kenjogo, Sonkeigo): most formal
                2 (Teineigo): medium, used daily
                1 (Futuskei): informal, casual
        """
        stem_orig, stem_no_h,root_last, pos_last = self._get_root_stem(text=text)
        stem = stem_orig if formality_level == 3 else stem_no_h

        # PRONOUNS: genders and neutralizes
        base_stem, has_pron = self._add_replace_remove_pronoun(stem=stem, root_last=root_last, pos_last=pos_last, formality_level=formality_level, gender=gender)
        
        # ENDING PARTICLES: genders for formality level 1
        g_particle = ""
        if formality_level == 1:
            particles = self.GENDER_MAP["particles"].get(gender, [])
            if particles:
                g_particle = particles[0]
        
        # pos_last is the full one for the gendering
        pos_last = pos_last[0]
        
        # VERBS
        if pos_last == "動詞":
            verb_stem = self._get_verb_stem(root=root_last)
            # For F1: select verbs that can use -nnda suffix instead of -ka
            is_nnda = any(w in base_stem for w in ["どう", "なぜ", "何で", "どうして", "いかに", "どんな", "時", "場合"])
            if formality_level == 1:
                if gender != "neutral":
                    if gender == "male":
                        if is_nnda or (root_last not in self.objective_roots):
                            return f"{base_stem}{root_last}んだ？"
                        else: return f"{base_stem}{root_last}か？"
                    else: return f"{base_stem}{root_last}{g_particle}？"
                else: # neutral
                    return f"{base_stem}{root_last}？"
            elif formality_level == 2:
                return f"{base_stem}{verb_stem}ますか？"
            elif formality_level == 3:
                if root_last in self.HONORIFIC_EXCEPTIONS:
                    return f"{base_stem}{self.HONORIFIC_EXCEPTIONS[root_last]}か？"
                else:
                    return f"{base_stem}お{verb_stem}になりますでしょうか？"
        
        # ADJ
        elif pos_last == "形容詞":
            if formality_level == 1:
                if gender != "neutral":
                    if gender == "male":
                        return f"{base_stem}{root_last}か？"
                    else:
                        return f"{base_stem}{root_last}{g_particle}？"
            elif formality_level == 2:
                return f"{base_stem}{root_last}ですか？"
            elif formality_level == 3:
                return f"{base_stem}{root_last}のでしょうか？" # のでしょうか いらっしゃいますでしょう
            
        # N, PRON, Na-ADJ, SUFFIXES (e.g. 的、派)
        elif pos_last in ["名詞", "代名詞", "形状詞", "接尾辞", "副詞"]:
            if formality_level == 1:
                if gender != "neutral":
                    if gender == "male":
                        if pos_last == "代名詞":
                            return f"{base_stem}{root_last}だ？"
                        else: return f"{base_stem}{root_last}か？"
                    else: return f"{base_stem}{root_last}{g_particle}？"
                else:
                    return f"{base_stem}{root_last}？"
            elif formality_level == 2:
                return f"{base_stem}{root_last}ですか？"
            elif formality_level == 3:
                if root_last in self.HONORIFIC_EXCEPTIONS:
                    return f"{base_stem}{self.HONORIFIC_EXCEPTIONS[root_last]}か？"
                else:
                    return f"{base_stem}{root_last}でしょうか？"
         
    def convert_all(self, text:str):
        """
            Takes a string of any formality level, and converts it to all 3 defined formality levels.
            Returns a dictionary with all 3 converted strings.
        """
        return [
                # neutral
                {"question": self.convert_specific(text, 3), "topic": "", "formality": "3", "gender": "neutral"}, 
                {"question":self.convert_specific(text, 2), "topic": "", "formality": "2", "gender": "neutral"},
                {"question": self.convert_specific(text, 1), "topic": "", "formality": "1", "gender": "neutral"},
                # gendered
                {"question": self.convert_specific(text, 3, "male"), "topic": "", "formality": "3", "gender": "male"},
                # NOTE: formality 3 is technically genderless cuz pronouns are the same
                {"question": self.convert_specific(text, 3, "male"), "topic": "", "formality": "3", "gender": "female"},
                {"question":self.convert_specific(text, 2, "male"), "topic": "", "formality": "2", "gender": "male"},
                {"question":self.convert_specific(text, 2, "female"), "topic": "", "formality": "2", "gender": "female"},
                {"question": self.convert_specific(text, 1, "male"), "topic": "", "formality": "1", "gender": "male"},
                {"question":self.convert_specific(text, 1, "female"), "topic": "", "formality": "1", "gender": "female"},
       
        ]







# examples = [
#     # Business & Career
#     "上司と意見が合わない時、どう伝える？",
#     "会議で発言するタイミングをどう見極める？",
#     "仕事のやる気が出ない時、どう対処する？",
#     "職場での人間関係を円滑にするコツは何？",
#     "転職を考える際に最も重視する条件は何？",
#     "効果的なプレゼンテーションの仕方は何？",
#     "お仕事とプライベートのバランスをどう取る？",
#     "ミスをしてしまった時の謝罪の仕方はどうする？",
#     "新入社員にアドバイスをするなら何を伝える？",
#     "自身のスキルアップのために、どのような学習を継続する？",
#     "効率的にタスクを管理する方法は何？",
#     "難しいプロジェクトを任されたらどう反応する？",
#     "リーダーシップを発揮するために必要な要素は何？",
#     "職場でのストレスをどう発散する？",
#     "リモートワークと出社、どちらが効率的？",
#     "電話対応で気をつけるべきマナーは何？",
#     "キャリアプランを立てる際に必要な視点は何？",
#     "お給与交渉をする適切なタイミングはいつ？",
#     "チーム内のコミュニケーションを活発にするにはどうする？",
#     "プレゼン資料を作る際、一番時間をかける場所はどこ？",

#     # Lifestyle & Preferences
#     "初心者におすすめの料理は何？",
#     "週末に一人で過ごすならどこへ行く？",
#     "朝食はパンとご飯のどちらが好き？",
#     "旅行に行くなら海と山のどちらを選ぶ？",
#     "最近買ってよかったと思うものは何？",
#     "料理を失敗したとき、どうやってリカバリーする？",
#     "おすすめの気分転換の方法は何？",
#     "毎日欠かさず続けている習慣は何？",
#     "コーヒーと紅茶、どちらをよく飲む？",
#     "映画館で映画を見るのは好き？",
#     "掃除を効率よく終わらせるコツは何？",
#     "理想的な休日の過ごし方は何？",
#     "犬派か、それとも猫派？",
#     "眠れない夜はどうやって過ごす？",
#     "外食と自炊、どちらの頻度が高い？",
#     "子供の頃に好きだった遊びは何？",
#     "お土産に喜ばれるお菓子は何？",
#     "部屋のインテリアでこだわっている点は何？",
#     "最近読んだ本の中で面白かったものは何？",
#     "冬に食べたくなる料理は何？",

#     # Trivia & General Knowledge
#     "日本で一番高い山の名前は何？",
#     "ラベンダーで有名な北海道の都市はどこ？",
#     "太陽系の中で一番大きい惑星は何？",
#     "徳川家康が江戸幕府を開いたのは何年？",
#     "水の沸点は摂氏何？",
#     "ブラジルの首都はどこ？",
#     "夏目漱石の代表作は何？",
#     "元素記号「H」が表す物質は何？",
#     "五輪旗にある輪の数はいくつ？",
#     "世界で一番長い川はどこ？",
#     "寿司の「シャリ」とは何？",
#     "富士山はどの県にまたがっている？",
#     "ピアノの鍵盤は全部でいくつある？",
#     "日本で最初に世界文化遺産に登録された場所はどこ？",
#     "1年は秒に換算するといくつ？",
#     "哺乳類の中で空を飛べるものは何？",
#     "法律を作る役割を持つ機関はどこ？",
#     "ダイヤモンドは何の元素からできている？",
#     "121の平方根は何？",
#     "シェイクスピアはどこの国の劇作家？",

#     # Philosophical & Abstract
#     "人生において一番大切なものは何？",
#     "幸せの定義とは何？",
#     "嘘をつくことが許されるのはどのような時？",
#     "過去に戻れるなら、いつに戻って何をしたい？",
#     "人間が生きる目的は何？",
#     "許せない事象に直面した時、どう心を整理する？",
#     "死後の世界はあると信じる？",
#     "お金で買えない価値があるものは何？",
#     "運命は自分で変えられるもの？",
#     "他人のために自分を犠牲にすることは正しい？",
#     "科学の進歩は常に人類を幸せにする？",
#     "孤独を感じた時、どうやって向き合う？",
#     "平和な世界を作るために個人ができることは何？",
#     "成功することと、誠実であること、どちらを選ぶ？",
#     "自分が死んだ後、どのように記憶される？",
#     "自由とは、どのような状態のことを指す？",
#     "失敗から学べる最も重要な教訓は何？",
#     "愛とは感情、それとも決意？",
#     "技術が発達しても変わらない人間の本質は何？",
#     "人生をどう生きればいい？",

#     # Weather & Daily Observations
#     "今日の空、雲が多い？",
#     "明日の午後は晴れる？",
#     "この辺りの冬はいつも乾燥する？",
#     "朝起きてから、一番最初に口にするものは何？",
#     "今の気分を色に例えるなら、何色っぽい？",
#     "最近、夜はよく眠れる？",
#     "今日は少し湿気が多い？",
#     "明日は傘を持って出かける？",
#     "この近くにおすすめのカフェはある？",
#     "今日は何時ごろに帰宅する？",
#     "最近、何か面白いニュースはある？",
#     "今週の土日は天気がいい？",
#     "お外の気温はどれくらい？",
#     "季節の中で春が一番好き？",
#     "街路樹の葉が落ちる？",
#     "今日は上着が必要なほど寒い？",
#     "お昼ご飯はもう食べる？",
#     "この道を通るのは初めて？",
#     "予報では夜から雨が降る？",
#     "今日は穏やかな天気？"
# ]

examples = [
    "上司と意見が合わない時、どう伝える？",
    "初心者におすすめの料理は何？",
    "シェイクスピアはどこの国の劇作家？",
    "人生をどう生きればいい？",
    "お外の気温はどれくらい？"
]

engine = FormalityConverter()
headers = ["q_id", "question", "topic", "formality", "gender"]
with open("check_final.txt", "w") as f:
    for i in examples:
        # q = engine.convert_specific(i, 1, "male")
        q = engine.convert_all(i)
        
        f.write(f'\n------------------------\nOriginal: {i}\n')
        # print(f'\n------------------------\nOriginal: {i}')
        # print(q)
        for k in q:
            # print(f"{k['formality']}, {k['gender']}: {k['question']}")
            f.write(f"{k['formality']}, {k['gender']}: {k['question']}\n")

# check =  "最近、夜はよく眠れる？"
# q = engine.convert_all(check)
# print(f'\n------------------------\nOriginal: {check}')
# for k in q:
#     print(f"{k['formality']}, {k['gender']}: {k['question']}")
# # print(engine._get_verb_stem(root="取る"))