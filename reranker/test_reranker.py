from reranker import LocalBookReranker

def test_reranker():
    # Initialize reranker
    reranker = LocalBookReranker(
        user_embeddings_path="../embeddings/gmf_user_embeddings.parquet",
        book_embeddings_path="../embeddings/gmf_book_embeddings.parquet",
        sbert_embeddings_path="../embeddings/SBERT_embeddings.parquet"
    )
    
    # Test with a specific user ID
    test_user_id = "99819"  # Using the user ID from your example
    recommendations = reranker.get_recommendations(test_user_id)
    
    print(f"\nFinal recommendations for user {test_user_id}:")
    print(recommendations)

if __name__ == "__main__":
    test_reranker() 