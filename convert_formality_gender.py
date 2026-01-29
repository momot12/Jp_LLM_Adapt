import csv
import re
from sudachipy import dictionary, tokenizer


# TODO (28.1.26): fix honorific issues on _apply_honorifics()
# TODO (later): add gender conversion

### NEUTRAL ### 
# 1: Normalize
# 2: Morphological analysis
# 3: Map respective formality suffix

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
                "male":   [],
                "female": ["かしら"],
                "neutral": ["の"]
            }
        }
        
        
    def _get_root_stem(self, text:str):
        """
            Private function. Morphological analysis.
            Takes any string and split into
            
                stem_orig: everything before last content word (no changes applied)
                stem_no_h: everything before last content word WITHOUT honorifics ('O'/'Go')
                   stem_h: everything before last content word WITH honorifics ('O'/'Go')
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
                                 if tokens[idx].part_of_speech()[0] in ["動詞", "名詞", "形容詞", "代名詞"]), 
                                 len(list(tokens))-1)                  
                      
        # LAST TOKEN is last content word
        last_token = tokens[idx_last_content]   
        # get dictionary form and pos of last word
        root_last = last_token.dictionary_form()    # '食べ' --> '食べる'
        # part_of_speech() gives ('動詞', '一般', '*', '*', '下一段-バ行', '連用形-一般')
        pos_last = last_token.part_of_speech()[0]   # e.g. '動詞' (verb) 
        
        
        # STEMS - original, without, and with honorifics
        stem_orig = "".join([t.surface() for t in token_list[:idx_last_content]])  # 'お昼ご飯は'
        stem_no_h = "".join([t.surface() for t in token_list[:idx_last_content]    # '昼ご飯は'
                            if t.part_of_speech()[0] != "接頭辞"])
        # TODO: check later
        stem_h = stem_orig#self._apply_honorific(tokens=token_list[:idx_last_content])       # 'お昼ご飯は'

        return stem_orig, stem_no_h, stem_h, root_last, pos_last
    
    def _get_honorific(self, token:str):
        """
            Decides which honorific form is prepended to the noun.
                A. Wago  (native Jp): 'O'
                    - includes Hiragana
                    - single char Kanji
                    - usually default
                B. Kango (sino-Jp)  : 'Go'
                    - >1 Kanjis 
        """
        o = "お"
        go = "ご"
        
        # Special cases: skip alphabet, Katakana, and numbers
        if re.search(r'[a-zA-Zア-ンー0-9０-９]', token):
            return ""
        # Special cases: skip single char (non-Kanji)
        if len(token) == 1 and not re.match(r'[\u4e00-\u9faf]', token):
            return ""
        
        # A. Wago
        if re.search(r'[ぁ-ん]', token) or len(token)==1:
            return o
        
        # B. Kango
        # regex: Common CJK Unified; \u4e0=first Kanji, \u9faf=last Kanji
        if re.fullmatch(r'[\u4e00-\u9faf]+', token):
            return go
        
        # fallback: return no honorific
        # NOTE: change to o?
        return ""
    
    def _apply_honorific(self, tokens):
        processed_with_h = []
        for i, t in enumerate(tokens):
            token = t.surface()
            pos_full = t.part_of_speech()
            pos_main = pos_full[0]
            pos_sub = pos_full[1]
            pos_subsub = pos_full[2]

            # keep existing honorifics
            if pos_main == "接頭辞":
                processed_with_h.append(token)
                continue
            
            # TODO: fix ご寿司 and ご元素記号
            # check if noun can have honorific (skips PROP, TIME nouns, PRON)
            is_eligible_noun = (pos_main == "名詞" and pos_sub == "普通名詞" and pos_subsub == "一般")
            is_mid_compound = (i > 0 and tokens[i-1].part_of_speech()[0] == "名詞")
            if is_eligible_noun:
                # check if pos of the token before noun is honorific
                has_honorific = (i > 0 and tokens[i-1].part_of_speech()[0] == "接頭辞")
                
                # add honorific if conditions met
                if is_eligible_noun and not is_mid_compound:
                    honorific = self._get_honorific(token)
                    processed_with_h.append(str(honorific)+str(token))
                else:
                    processed_with_h.append(token)
            
            # not noun, just append
            else:
                processed_with_h.append(token)
        
        return "".join(processed_with_h)
    
    def _get_verb_stem(self, root:str):
        """Converts dictionary form to masu-stem (Ren'youkei)."""
        # Irregular verbs: suru -> shi
        if root == "する": return "し"
        # kuru -> ki (Kanji & Hiragana)
        if root == "来る": return "来"
        if root == "くる": return "く"
        
        # Ru-Verbs (Ichidan): remove 'ru'
        # Note: In a research context, you'd check POS subtype for 'Ichidan'
        if root.endswith("る"):
            # Simple check: if the char before 'ru' is in 'i' or 'e' row
            return root[:-1]
            
        # U-Verbs (Godan): Change last char from 'u' to 'i' row
        vowels = {'う': 'い', 'く': 'き', 'ぐ': 'ぎ', 'す': 'し', 'つ': 'ち', 'ぬ': 'に', 'ふ': 'ひ', 'む': 'み', 'る': 'り'}
        last_char = root[-1]
        if last_char in vowels:
            return root[:-1] + vowels[last_char]
        return root
    
    def convert_specific(self, text:str, formality_level:int, gender:str="neutral"):
        """
            Takes a string of any formality level and converts it into targeted formality level.
            
            Defined formality levels:
                3 (Kenjogo, Sonkeigo): most formal
                2 (Teineigo): medium, used daily
                1 (Futuskei): informal, casual
        """
        stem_orig, stem_no_h, stem_h, root_last, pos_last = self._get_root_stem(text=text)
        base_stem = stem_h if formality_level == 3 else stem_no_h
        
        # VERBS
        if pos_last == "動詞":
            verb_stem = self._get_verb_stem(root=root_last)
            if formality_level == 1:
                return f"{base_stem}{root_last}？"
            elif formality_level == 2:
                return f"{base_stem}{verb_stem}ますか？"
            elif formality_level == 3:
                return f"{base_stem}お{verb_stem}になられますか？"
        
        # ADJ
        elif pos_last == "形容詞":
            if formality_level == 1:
                return f"{base_stem}{root_last}？"
            elif formality_level == 2:
                return f"{base_stem}{root_last}ですか？"
            elif formality_level == 3:
                return f"{base_stem}{root_last}のでしょうか？"
            
        # N, PRON
        elif pos_last in ["名詞", "代名詞"]:
            if formality_level == 1:
                return f"{base_stem}{root_last}？"
            elif formality_level == 2:
                return f"{base_stem}{root_last}ですか？"
            elif formality_level == 3:
                return f"{base_stem}{root_last}でしょうか？"
         
    def convert_all(self, text):
        """
            Takes a string of any formality level, and converts it to all 3 defined formality levels.
            Returns a dictionary with all 3 converted strings.
        """
        return {
            "level 1": self.convert_specific(text, 1),
            "level 2": self.convert_specific(text, 2),
            "level 3": self.convert_specific(text, 3)
        }




engine = FormalityConverter()


examples = ['元素記号「H」が表す物質は何？',
            '五輪旗にある輪の数はいくつ？',
            '世界で一番長い川はどこ？',
            '寿司の「シャリ」とは何のこと？',
            'お昼ご飯は食べましたか？',
            '人生をどう生きればいいですか？']

text_1 = 'お昼ご飯は食べる？'
text_2 = 'お昼ご飯は食べますか？'
text_3 = '何が良い？'
text_4 = 'シェイクスピアはどこの国の劇作家？'


# engine._get_root_stem(text=text_4)

# a = engine.convert_all(text=text_4)
# b = engine.convert_all(text=text_2)

# print(f'\n{a}\n\n{b}')
# print(f'{a}')
for i in examples:
    conv = engine.convert_all(text=i)
    for k, v in conv.items():
        print(f"{k}: {v}")
    print()
