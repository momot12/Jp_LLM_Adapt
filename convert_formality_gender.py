import csv
from sudachipy import dictionary
from sudachipy import tokenizer

# original_data = "/mount/arbeitsdaten/studenten4/takamamo/sample_data.csv"


# def read_csv(data_path:str):
#     with open(data_path, 'r') as f:
#         data = f.read()
    
#     return data

# print(read_csv(original_data))


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
        
        
    def _get_root_stem(self, text:str):
        """
            Private function. Morphological analysis.
            Takes any string and split into
            
                     stem: everything before the last word
                root_last: dictionary form of the last word
                 pos_last: part-of-speech of the last word
                
            to convert endings into arbitrary formality form.
        """
        # STRIP text from ?, ending particle, or white space
        # e.g. お昼ご飯は食べた? --> お昼ご飯は食べた
        stripped_text = text.rstrip("？? か")
        
        # TOKENIZE -> tokens type:sudachipy.morphemelist.MorphemeList
        # e.g. お昼ご飯は食べた --> お 昼ご飯 は 食べ た
        tokens = self.tokenizer.tokenize(text=stripped_text, mode=self.mode)
        
        # tokens = []
        # has_honorific = False
        # for t in tokenized_raw:
        #     # check POS='接頭辞' (prefix) and drop it
        #     if t.part_of_speech()[0] == "接頭辞":
        #         continue
        #     tokens.append(t)
        
        # fallback if nothing tokenized
        if not tokens: return None, None, None, None
        
        # STEM_orig and get LAST TOKEN 
        token_list = list(tokens)                      # list of <Morpheme> obj)
        # next(iterator, default) -> last token default
        idx_last_content = next((idx for idx in range(len(list(tokens)-1, -1, -1))
                                 if tokens[idx].part_of_speech()[0] if pos in ["動詞", "名詞", "形容詞", "代名詞"]), 
                                 len(list(tokens))-1)                  
        
        # len(token_list) - 1          # default is last token, e.g  <Morpheme(た, 7:8, (0, 83557))>
        # for idx in range(len(token_list)-1, -1, -1):
        #     pos = tokens[idx].part_of_speech()[0]
        #     # check if pos is V, Adj, N, or PRON
        #     if pos in ["動詞", "名詞", "形容詞", "代名詞"]:
        #         idx_last_content = idx
        #         break
                                  
        # LAST TOKEN is last content word
        last_token = tokens[idx_last_content]   
        # get dictionary form and pos of last word
        root_last = last_token.dictionary_form()    # '食べ' --> '食べる'
        # part_of_speech() gives ('動詞', '一般', '*', '*', '下一段-バ行', '連用形-一般')
        pos_last = last_token.part_of_speech()[0]   # e.g. '動詞' (verb) 
        
        
        # STEMS - original, with & without honorifics
        stem_orig = "".join([t.surface() for t in token_list[:idx_last_content]])  # 'お昼ご飯は'
        stem_no_h = "".join([t.surface() for t in token_list[:idx_last_content]    # '昼ご飯は'
                           if t.part_of_speech()[0] != "接頭辞"])
        stem_h = None
        
        
                                      
        # surface() takes た of  <Morpheme(た, 7:8, (0, 83557))>
        # stem = "".join([t.surface() for t in token_list[:idx_last_content]])      # 'お昼ご飯は'
        

        return stem_no_h, stem_h, root_last, pos_last
    
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
    
    def convert_specific(self, text:str, formality_level:int):
        """
            Takes a string of any formality level and converts it into targeted formality level.
            
            Defined formality levels:
                3 (Kenjogo, Sonkeigo): most formal
                2 (Teineigo): medium, used daily
                1 (Futuskei): informal, casual
        """
        stem, root_last, pos_last = self._get_root_stem(text=text)
        
        # VERBS
        if pos_last == "動詞":
            verb_stem = self._get_verb_stem(root=root_last)
            # Level 1:
            if formality_level == 1:
                return f"{stem}{root_last}？"
            # Level 2:
            elif formality_level == 2:
                return f"{stem}{verb_stem}ますか？"
            # Level 3:
            elif formality_level == 3:
                return f"{stem}お{verb_stem}になられますか？"
        
        # ADJ
        elif pos_last == "形容詞":
            if formality_level == 1:
                return f"{stem}{root_last}？"
            elif formality_level == 2:
                return f"{stem}{root_last}ですか？"
            elif formality_level == 3:
                return f"{stem}{root_last}のでしょうか？"
            
        # N, PRON
        elif pos_last in ["名詞", "代名詞"]:
            if formality_level == 1:
                return f"{stem}{root_last}？"
            elif formality_level == 2:
                return f"{stem}{root_last}ですか？"
            elif formality_level == 3:
                return f"{stem}{root_last}でしょうか？"
         
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
            '寿司の「シャリ」とは何のこと？']

text_1 = 'お昼ご飯は食べた？'
text_2 = 'お昼ご飯は食べましたか？'

text_3 = '何が良い？'


engine._get_root_stem(text=text_1)

# a = engine.convert_all(text=text_3)
# b = engine.convert_all(text=text_2)

# print(f'\n{a}\n\n{b}')
# # print(f'{a}')
# for i in examples:
#     conv = engine.convert_all(text=i)
#     print(f"{conv}\n")


## check if O, go there.
## in level 1, 2 remove the o, go 
## function to add O, go where suitable