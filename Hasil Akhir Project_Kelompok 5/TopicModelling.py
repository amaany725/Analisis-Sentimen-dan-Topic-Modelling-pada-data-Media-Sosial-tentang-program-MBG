#AMAANY (23031554148)
#TSALMA MASYTHA ZAHWA (23031554103)
#NURIN NASI'AH SALSABILA (23031554148)

import ast
import pandas as pd
from gensim.models import Phrases, LdaModel, CoherenceModel
from gensim.models.phrases import Phraser
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def main():
    # 1️⃣ Load data
    df = pd.read_csv("clean_review (2).csv")
    texts = df['full_text_stemmed'].apply(ast.literal_eval).tolist()

    # 2️⃣ Bigram dan trigram
    bigram = Phrases(texts, min_count=5, threshold=100)
    trigram = Phrases(bigram[texts], threshold=100)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    texts_bigrams = [bigram_mod[doc] for doc in texts]
    texts_trigrams = [trigram_mod[bigram_mod[doc]] for doc in texts]

    # 3️⃣ Dictionary dan corpus
    dictionary = corpora.Dictionary(texts_trigrams)
    dictionary.filter_extremes(no_below=5, no_above=0.9)
    corpus = [dictionary.doc2bow(text) for text in texts_trigrams]

    # 4️⃣ Word Cloud & Top 10 kata
    joined_trigrams = [' '.join(doc) for doc in texts_trigrams]
    vectorizer = CountVectorizer()
    bow_encoded = vectorizer.fit_transform(joined_trigrams)

    feature_names = vectorizer.get_feature_names_out()
    bow_scores = bow_encoded.toarray().sum(axis=0)
    bow_dict = dict(zip(feature_names, bow_scores))

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bow_dict)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud dari N-Grams (BoW)')
    plt.show()

    top_10_words = sorted(bow_dict.items(), key=lambda item: item[1], reverse=True)[:10]
    words, scores = zip(*top_10_words)

    plt.figure(figsize=(10, 6))
    plt.barh(words, scores, color='orange')
    plt.title('Top 10 Kata Terpopuler (N-Grams BoW)')
    plt.xlabel('Frekuensi')
    plt.gca().invert_yaxis()

    for i, v in enumerate(scores):
        plt.text(v + 0.5, i, str(v), color='black', va='center')

    plt.tight_layout()
    plt.show()

    # 5️⃣ LDA Model
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=5,
                         passes=10,
                         random_state=42)

    topics = lda_model.print_topics(num_words=10)
    for idx, topic in topics:
        print(f"\n🧠 Topik {idx}:")
        print(topic)

    # Dokumen ke-topik dominan
    topic_per_doc = [max(lda_model.get_document_topics(doc), key=lambda x: x[1])[0] for doc in corpus]
    df['topik_lda'] = topic_per_doc
    print(df[['clean_review', 'topik_lda']].head())

    # 6️⃣ Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=texts_trigrams,
                                         dictionary=dictionary,
                                         coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print(f"\n🧠 Coherence Score (5 topik): {coherence_lda:.4f}")

    # 7️⃣ Uji coherence beberapa topik
    coherence_values = []
    topic_range = range(2, 11)

    for num_topics in topic_range:
        model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         passes=10,
                         random_state=42)

        coherence_model = CoherenceModel(model=model,
                                         texts=texts_trigrams,
                                         dictionary=dictionary,
                                         coherence='c_v')

        score = coherence_model.get_coherence()
        coherence_values.append(score)
        print(f"Topik: {num_topics} → Coherence Score: {score:.4f}")

    # 8️⃣ Final LDA dengan topik terbaik
    best_lda = LdaModel(corpus=corpus,
                        id2word=dictionary,
                        num_topics=8,
                        passes=10,
                        random_state=42)

    topics = best_lda.print_topics(num_words=10)
    for idx, topic in topics:
        print(f"\n🧠 Topik {idx}:")
        print(topic)

    # 9️⃣ Visualisasi pyLDAvis
    pyldavis_prepared = gensimvis.prepare(best_lda, corpus, dictionary)
    pyLDAvis.display(pyldavis_prepared)
    pyLDAvis.save_html(pyldavis_prepared, 'output.html')
    print("\n✅ File output.html sudah dibuat untuk visualisasi pyLDAvis!")

if __name__ == "__main__":
    main()
