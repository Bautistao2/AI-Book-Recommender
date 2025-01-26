import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout

def create_recommendation_model(num_users, num_books, embedding_dim=50):
    """
    Creates a recommendation model using embeddings for users and books.

    Args:
        num_users (int): Total number of unique users.
        num_books (int): Total number of unique books.
        embedding_dim (int): Dimension of the embedding vector for users and books.

    Returns:
        keras.Model: Compiled recommendation model.
    """
    # Input layer for user IDs
    user_input = Input(shape=(1,), name="user_input")
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name="user_embedding")(user_input)
    user_vector = Flatten()(user_embedding)

    # Input layer for book IDs
    book_input = Input(shape=(1,), name="book_input")
    book_embedding = Embedding(input_dim=num_books, output_dim=embedding_dim, name="book_embedding")(book_input)
    book_vector = Flatten()(book_embedding)

    # Concatenate user and book embeddings
    concatenated = Concatenate()([user_vector, book_vector])

    # Dense layers for learning interactions
    # The dense layer with 128 units strikes a balance between model complexity and computational efficiency.
    # A higher number of units allows the model to learn more complex patterns, but it may also increase the risk of overfitting.
    dense_1 = Dense(128, activation="relu", name="dense_1")(concatenated)

    # Dropout with a rate of 0.3 helps prevent overfitting by randomly setting 30% of the layer's units to zero during training.
    # This value strikes a balance between regularization and retaining enough information for learning.
    dropout_1 = Dropout(0.3, name="dropout_1")(dense_1)

    dense_2 = Dense(64, activation="relu", name="dense_2")(dropout_1)
    dropout_2 = Dropout(0.3, name="dropout_2")(dense_2)
    output = Dense(1, activation="linear", name="output")(dropout_2)  # Final prediction layer

    # Compile the model
    model = Model(inputs=[user_input, book_input], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    return model
