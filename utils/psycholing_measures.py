import spacy
# import elfen
# import polars as pl


def get_psycholinguistic_metrics(sentence):
    # load the model you already have
    model_name = "ja_core_news_sm"
    nlp = spacy.load(model_name) # tokenization, POS-tagging (sudachi dict)
    doc = nlp(sentence)
    
    ## PARTICLE USAGE
    # more particles == more formal
    target_particles = ["は", "を", "に", "が", "の", "で", "と"]
    particle_counts = {p: 0 for p in target_particles}
    particle_counts["others"] = 0
    
    total_particles = 0
    for token in doc:
        if token.pos_ == "ADP":                 # universal tag for particles
            total_particles += 1
            if token.text in particle_counts:
                particle_counts[token.text] += 1
            else:
                particle_counts["others"] += 1

    ## LEXICAL: Kango (complex) vs. Wago (simple)
    content_tokens = [t for t in doc if t.pos_ in ["NOUN", "VERB", "ADJ", "PROPN", "ADV"]]
    # Kanji combo that is more than 2 kanjis == Kango
    kango_count = sum(1 for t in content_tokens if 
                  any('\u4e00' <= c <= '\u9faf' for c in t.text) and 
                  not any('\u3040' <= c <= '\u309f' for c in t.text) and
                  len(t.text) > 1)
    # 0.0 (simple) ~ 1.0 (complex)
    lex_complexity = round(kango_count / len(content_tokens), 2) if content_tokens else 0

    ## SYNTACTIC: sub-clauses
    def get_tree_depth(node):
        if not list(node.children): return 1
        return 1 + max(get_tree_depth(child) for child in node.children)
    
    roots = [token for token in doc if token.head == token]
    depth = max([get_tree_depth(root) for root in roots]) if roots else 0

    ## AoA PROXY: more Kanji == more complex
    kanji_chars = sum(1 for c in sentence if '\u4e00' <= c <= '\u9faf')
    # 5 = lower bound, max = 15.0, *10 to make it a range; rounded 
    aoa_proxy = round(5 + (kanji_chars / len(sentence)) * 10, 1)

    return {
        "AoA": aoa_proxy, 
        "syntax_depth": depth, 
        "lexical_complexity": lex_complexity,
        "total_count_particle": total_particles,
        # "particle_analysis": {
        #      "details": particle_counts
        # }
    }


# # Test Examples
# sentences = [
#     "おにぎりを食べた。", # Simple
#     "高度な経済政策を導入することにより、市場の活性化を図る。" # Complex
# ]

# for s in sentences[1:]:
#     print(f"Sentence: {s}")
#     print(get_psycholinguistic_metrics(s))
#     print("-" * 30)