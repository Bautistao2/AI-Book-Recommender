import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Function to load the datasets
def load_data(ratings_path, books_path):
    """
    Load ratings and books data from file paths.

    Args:
        ratings_path (str): Path to the ratings CSV file.
        books_path (str): Path to the books CSV file.

    Returns:
        ratings (DataFrame): DataFrame containing user ratings.
        books (DataFrame): DataFrame containing book metadata.
    """
    ratings = pd.read_csv(ratings_path)
    books = pd.read_csv(books_path)
    return ratings, books

# Function to preview the data
def preview_data(ratings, books):
    """
    Print the first few rows of the ratings and books data.

    Args:
        ratings (DataFrame): DataFrame containing user ratings.
        books (DataFrame): DataFrame containing book metadata.
    """
    print("Ratings DataFrame:")
    print(ratings.head())
    print("\nBooks DataFrame:")
    print(books.head())

# Function to preprocess data
def preprocess_data(ratings, books):
    """
    Preprocess ratings and books data.

    Args:
        ratings (DataFrame): DataFrame containing user ratings.
        books (DataFrame): DataFrame containing book metadata.

    Returns:
        train (DataFrame): Training set.
        val (DataFrame): Validation set.
        book_metadata (DataFrame): Metadata of books.
        user_id_map (dict): Mapping of user IDs to indices.
        book_id_map (dict): Mapping of book IDs to indices.
    """
    # Encode user IDs and book IDs
    user_ids = ratings['user_id'].unique()
    book_ids = ratings['book_id'].unique()

    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    book_id_map = {book_id: idx for idx, book_id in enumerate(book_ids)}

    ratings['user_encoded'] = ratings['user_id'].map(user_id_map)
    ratings['book_encoded'] = ratings['book_id'].map(book_id_map)

    # Extract book metadata
    book_metadata = books[['book_id', 'title', 'authors']]

    # Split the dataset
    train, val = train_test_split(ratings, test_size=0.2, random_state=42)

    return train, val, book_metadata, user_id_map, book_id_map

# Function to create a user-book interaction matrix
def create_interaction_matrix(train):
    """
    Create a user-book interaction matrix where rows represent users and columns represent books.

    Args:
        train (DataFrame): Training set.

    Returns:
        np.array: Interaction matrix.
    """
    n_users = train['user_encoded'].nunique()
    n_books = train['book_encoded'].nunique()

    # Create a matrix of zeros
    interaction_matrix = np.zeros((n_users, n_books))

    # Fill the matrix with ratings
    for row in train.itertuples():
        interaction_matrix[row.user_encoded, row.book_encoded] = row.rating

    return interaction_matrix

# Function to recommend books for a specific user
def recommend_books(model, user_id, user_id_map, book_id_map, books, top_n=5):
    """
    Generate book recommendations for a specific user.

    Args:
        model: Trained recommendation model.
        user_id (int): User ID to generate recommendations for.
        user_id_map (dict): Mapping of original user IDs to encoded indices.
        book_id_map (dict): Mapping of original book IDs to encoded indices.
        books (DataFrame): DataFrame containing book metadata.
        top_n (int): Number of recommendations to generate.

    Returns:
        DataFrame: Recommended books with titles and authors.
    """
    if user_id not in user_id_map:
        print(f"User ID {user_id} not found in the dataset.")
        return pd.DataFrame()

    user_encoded = user_id_map[user_id]

    # Generate predictions for all books
    all_books_encoded = np.array(list(book_id_map.values()))
    user_array = np.array([user_encoded] * len(all_books_encoded))

    predictions = model.predict([user_array, all_books_encoded]).flatten()

    # Get top N book indices with highest predicted ratings
    top_indices = predictions.argsort()[-top_n:][::-1]
    recommended_book_ids = [list(book_id_map.keys())[i] for i in top_indices]

    # Filter and return book metadata
    return books[books['book_id'].isin(recommended_book_ids)][['title', 'authors']]
