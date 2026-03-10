import os
import csv
import spacy
# import elfen
## NOTE: after upgrading transformer from 4.40 to 5.2, this issue arose.
##       This update was due to a model that needed upgraded transformers.
##       THIS IS A QUICK FIX!!! TODO: check later again (23.2.26)
##          --> use old venv or keep quick fix (10.3.26)
import sys
# provide the missing path that Transformers 5.x removed
try:
    from transformers import tokenization_utils_base
    sys.modules['transformers.tokenization_utils'] = tokenization_utils_base
except ImportError:
    pass # Handle cases where transformers might not be installed yet


class PschoLingMeasures():
    def __init__(self, 
                 model_name:str="ja_core_news_sm",
                 aoa_list_path:str="/mount/arbeitsdaten/studenten4/takamamo/data_AoA.csv",
                ):
        # tokenization, POS-tagging (sudachi dict)
        self.loaded_model = spacy.load(model_name)

        # postposition definitions (particles)
        self.skippable_ADPs = ["を", "は", "が", "へ", "に"]
        # take all particles that are tagged as ADP in spacy -> ADP: universal tag for particles
        self.common_ADPs = ["の", "は", "に", "を", "が", "で", "と", "も", "から", "まで", 
                            "より", "へ", "か", "や", "など", "ほど", "ばかり", "だけ", 
                            "ぐらい", "くらい", "なり", "やら", "だの", "し", "ぜ", "わ"] 

        # process AoA rating list (to keep it efficient)
        self.aoa_lookup = {}
        if aoa_list_path and os.path.exists(aoa_list_path):
            # read csv and load specific columns
            with open(aoa_list_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.aoa_lookup = {row["Word"]: float(row["meanAoA"]) for row in reader}

    def count_particles(self, doc):
        """
            Counts skippable and overall number of particles. Lists of paricles are defined in init.
            Args:
                doc: a sentence processed in spacy model.
            Return: total counts of both lists.
        """
        # init count dicts
        all_adps = {p: 0 for p in self.common_ADPs}
        all_adps["others"] = 0
        skippable_adps = {p: 0 for p in self.skippable_ADPs}
        
        total_particles = 0
        total_skippable = 0
        
        for token in doc:
            # ADP: universal tag for particles
            if token.pos_ == "ADP":
                # all ADP             
                if token.text in all_adps:
                    total_particles += 1
                    all_adps[token.text] += 1
                else:
                    all_adps["others"] += 1

                # Skippable ADP
                if token.text in skippable_adps:
                    total_skippable += 1
                    skippable_adps[token.text] += 1
        
        print(f"Not counted particles: {all_adps['others']}")
                    
        return total_particles, total_skippable
    
    def get_lexical_complexity(self, doc, alpha=5):
        """
            Calculate lexical complexity - as a proxy: Kango (complex) vs. Wago (simple).
            Calculation: (#Kango/#Content_word + alpha)
            Alpha serves as a smoothing factor to penalize short outputs. 
            Args:
                doc: a sentence processed in spacy model.
            Return: Lexical complexity number between 0.0~1.0
        """
        content_tokens = [t for t in doc if t.pos_ in ["NOUN", "VERB", "ADJ", "PROPN", "ADV"]]
        # Kanji combo that is more than 2 kanjis == Kango
        kango_count = sum(1 for t in content_tokens if 
                    any('\u4e00' <= c <= '\u9faf' for c in t.text) and 
                    not any('\u3040' <= c <= '\u309f' for c in t.text) and
                    len(t.text) > 1)
        # 0.0 (simple) ~ 1.0 (complex)
        lex_complexity = round(kango_count / len(content_tokens) + alpha, 4) if content_tokens else 0
        return lex_complexity
    
    def get_syntactic_depth(self, doc):
        """
            Calculate syntactic complexity by using tree depth.
            Args:
                doc: a sentence processed in spacy model.
            Return: Syntactic complexity number between 1.0~8.0+
        """
        # recursively count
        def _get_tree_depth(node):
            if not list(node.children): return 1
            return 1 + max(_get_tree_depth(child) for child in node.children)
        
        # identify root == verb at the end
        roots = [token for token in doc if token.head == token]
        # uses SudachiDict, so noun compounds like 経済政策 are split -> 経済 depends on 政策
        depth = max([_get_tree_depth(root) for root in roots]) if roots else 0
        return depth

    def get_AoA_proxy_list(self, sentence, doc):
        """
            Calculate Age of Acquisition (AoA) using a scaled proxy (more Kanji==more complex)
            and based on a list (Mochizuki & Ota, 2025). If the word is not in the list, it is skipped.
            
            Args:
                sentence: model's first output sentence.
            Return:
                - AoA proxy: 5.0~15.0+
                - AoA list: 1~7
        """
        # AoA proxy calc
        kanji_chars = sum(1 for c in sentence if '\u4e00' <= c <= '\u9faf')
        # 5 = lower bound, max = 15.0, *10 to make it a range; rounded 
        aoa_proxy = round(5 + (kanji_chars / len(sentence)) * 10, 1)
        
        # AoA based on list 
        aoa_scores = []
        for token in doc:
            # get lemma and the word itself
            lemma = token.lemma_
            text = token.text
            
            if lemma in self.aoa_lookup:
                aoa_scores.append(self.aoa_lookup[lemma])
            elif text in self.aoa_lookup:
                aoa_scores.append(self.aoa_lookup[text])
        
        aoa_list = round(sum(aoa_scores) / len(aoa_scores), 2) if aoa_scores else 0.0

        return aoa_proxy, aoa_list
    
    def get_psycholinguistic_metrics(self, sentence):
        """
            Calculates psycholinguistic metrics based on first sentence of model output.
            Metrics:
                - Age of Acquisition (AoA): once proxy calculation, once based on an external list
                - Syntactic complexity: tree depth
                - Lexical complexity: proxy calculation with #Kango (more==complex)
                - #Particles: once skippable particles, once common particles (ADP in SpacCy)
        """
        # load the model
        doc = self.loaded_model(sentence)
        
        ## PARTICLE USAGE
        # more particles == more formal
        total_particles, total_particles_skip = self.count_particles(doc=doc)

        ## LEXICAL: Kango (complex) vs. Wago (simple)
        lex_complexity = self.get_lexical_complexity(doc=doc, alpha=5)

        ## SYNTACTIC: sub-clauses
        depth = self.get_syntactic_depth(doc=doc)
        # for debug
        if depth <= 0:
            print(f"[DEPTH ISSUE]: {depth}\n{sentence}")
    

        ## AoA PROXY: more Kanji == more complex
        aoa_proxy, aoa_list = self.get_AoA_proxy_list(sentence=sentence, doc=doc)

        return {
            "AoA": aoa_proxy, 
            "AoA_w_list": aoa_list,
            "syntax_depth": depth, 
            "lexical_complexity": lex_complexity,
            "total_count_particle": total_particles,
            "count_particle_skip": total_particles_skip,
            # "particle_analysis": {
            #      "details": particle_counts
            # }
        }


# # Test Examples
# sentences = [
#     "おにぎりを食べた。", # Simple
#     "高度な経済政策を導入することにより、市場の活性化を図る。" # Complex
# ]

# nlp = PschoLingMeasures()
# for s in sentences:
#     print(f"\nSentence: {s}")
#     a = nlp.get_psycholinguistic_metrics(sentence=s)
#     for k,v in a.items():
#         print(f"{k}: {v}")
#     print("-" * 30)
