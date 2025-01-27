from utils import load_data, preview_data, preprocess_data, create_interaction_matrix
from model import create_recommendation_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# File paths for the datasets
ratings_path = 'data/ratings.csv'
books_path = 'data/books.csv'

# Step 1: Load data
data, books = load_data(ratings_path, books_path)

# Step 2: Preview the data (first 5 rows)
preview_data(data, books)

# Step 3: Preprocess data (encoding IDs, splitting data, etc.)
# The preprocess_data function handles the following:
# - Encodes user IDs and book IDs into unique numerical indices.
# - Splits the ratings dataset into training and validation sets.
# - Extracts metadata from the books dataset (e.g., titles and authors).
# - Returns mappings between original IDs and encoded indices
train, val, book_metadata, user_id_map, book_id_map = preprocess_data(data, books)

# Step 4: Create user-book interaction matrix
interaction_matrix = create_interaction_matrix(train)

# Step 5: Obtain the total number of unique users and books
num_users = len(user_id_map)
num_books = len(book_id_map)

# Step 6: Create the recommendation model
# The embedding dimension of 50 is chosen as a balance between model complexity and performance.
# Larger embedding dimensions can capture more features but may lead to overfitting or increased computational cost.
# Smaller dimensions may result in underfitting.
model = create_recommendation_model(num_users=num_users, num_books=num_books, embedding_dim=50)

# Print a summary of the model structure
model.summary()

# Step 7: Prepare training and validation datasets
train_user_input = train['user_encoded'].values
train_book_input = train['book_encoded'].values
train_ratings = train['rating'].values

val_user_input = val['user_encoded'].values
val_book_input = val['book_encoded'].values
val_ratings = val['rating'].values

# Step 8: Define callbacks for early stopping and model checkpointing
# EarlyStopping: Stops training when validation loss stops improving
# ModelCheckpoint: Saves the best model based on validation loss
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
]

# Step 9: Train the model with callbacks
history = model.fit(
    [train_user_input, train_book_input],  # Input: user and book IDs
    train_ratings,                        # Labels: ratings
    validation_data=([val_user_input, val_book_input], val_ratings),
    epochs=3,                            # Adjust epochs as needed
    batch_size=64,                        # Batch size
    verbose=1,                            # Verbosity level for logging
    callbacks=callbacks                   # Apply early stopping and checkpointing
)

# Step 10: Evaluate the model
# Perform predictions on the validation set
val_predictions = model.predict([val_user_input, val_book_input]).flatten()

# Calculate evaluation metrics
mae = mean_absolute_error(val_ratings, val_predictions)
rmse = np.sqrt(mean_squared_error(val_ratings, val_predictions))

print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

# Step 11: Generate recommendations
from utils import recommend_books

# Choose a user for recommendations
user_id_to_recommend = 1  # You can change this to any valid user ID

# Generate recommendations
print(f"\nGenerating recommendations for user {user_id_to_recommend}...")
recommended_books = recommend_books(
    model=model,
    user_id=user_id_to_recommend,
    user_id_map=user_id_map,
    book_id_map=book_id_map,
    books=books,
    top_n=5  # Number of recommendations
)

print("\nRecommended Books:")
print(recommended_books)

