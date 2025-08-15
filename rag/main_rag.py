import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MainRAG:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.top_categories = self.df['description'].drop_duplicates().to_list()
        self.vectorizer = TfidfVectorizer()
        self.category_vectors = self.vectorizer.fit_transform(self.top_categories)

    def classify_query(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.category_vectors)
        most_similar_category_index = similarities.argsort()[0][-3:][::-1]
        for idx in most_similar_category_index:
            yield self.top_categories[idx]

if __name__ == '__main__':
    main_rag = MainRAG('scraped_items_v2.csv')
    query = "Askerliğim"
    category = main_rag.classify_query(query)
    print(f"Query: '{query}'")
    print(f"Predicted Category: {list(category)}")