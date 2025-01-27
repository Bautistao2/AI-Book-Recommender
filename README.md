# 📚 Book Recommendation System

### **Overview**
Welcome to the Book Recommendation System! This project leverages deep learning to provide personalized book recommendations based on user preferences. By learning complex interactions between users and books, the system predicts ratings for unseen books and suggests the most relevant ones to each user. It's a hands-on demonstration of building and deploying a collaborative filtering-based recommendation system.

---

## **🚀 Features**
- **Deep Learning-Based Model**: A neural network with embeddings to represent users and books in a shared feature space.
- **Scalable Architecture**: Designed to handle thousands of users and books efficiently.
- **Interactive Web Interface**: Built using Streamlit for seamless user interaction.
- **Comprehensive Metrics**: Evaluation using MAE and RMSE to measure model accuracy.
- **Preprocessed Datasets**: Encoded user and book IDs for optimized training.

---

## **📂 Project Structure**
```
├── data/
│   ├── books.csv           # Metadata of books
│   ├── ratings.csv         # User-book ratings
├── main.py                 # Main script for training and recommendation
├── model.py                # Defines the recommendation model architecture
├── utils.py                # Helper functions for data loading and processing
├── app.py                  # Streamlit app for interactive recommendations
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## **💻 How It Works**

### **1. Data Loading**
- The system uses two datasets:
  - `ratings.csv`: Contains user ratings for books (user_id, book_id, rating).
  - `books.csv`: Metadata about books (book_id, title, authors).

### **2. Data Preprocessing**
- Encode user and book IDs into integers for neural network processing.
- Split the data into training and validation sets (80/20 split).

### **3. Model Architecture**
The recommendation model is built using TensorFlow and includes:
1. **Embeddings** for users and books to capture their latent features.
2. **Dense Layers** to learn complex interactions between users and books.
3. **Dropout Layers** to prevent overfitting.
4. **Output Layer** to predict the rating for a user-book pair.

### **4. Training and Evaluation**
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Metrics**: Mean Absolute Error (MAE)
- The model is trained to minimize MSE and validated using MAE and RMSE.

### **5. Recommendations**
- For a given user, the system predicts ratings for all books.
- Books with the highest predicted ratings are recommended.

### **6. Deployment**
- The system is deployed using **Streamlit**, offering an intuitive web interface for:
  - Inputting a user ID.
  - Viewing top book recommendations for the user.

---

## **📊 Results**
### **Model Performance**
- **MAE (Mean Absolute Error)**: 0.6387
- **RMSE (Root Mean Squared Error)**: 0.8333

These metrics indicate that the model provides accurate recommendations, with minimal deviation from actual user ratings.

---

## **📦 Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

### **2. Set Up the Environment**
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate    # For Windows
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
Launch the interactive recommendation system:
```bash
streamlit run app.py
```

---

## **🛠️ Technologies Used**
- **Python**: Core programming language.
- **TensorFlow/Keras**: For building and training the recommendation model.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: For splitting data into training and validation sets.
- **Streamlit**: Web-based deployment for interactive recommendations.
- **Matplotlib**: For data visualization (optional).

---

## **🌟 Future Improvements**
1. **Hyperparameter Tuning**:
   - Optimize embedding dimensions, dropout rates, and layer sizes.
2. **Cold Start Problem**:
   - Implement content-based filtering for new users or books.
3. **Incorporate Additional Features**:
   - Add genres, publication year, and user demographics.
4. **Scalability**:
   - Adapt the system to handle millions of users and books.

---

## **📧 Contact**
For questions, feedback, or collaboration, feel free to reach out:
- **Name**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## **📜 License**
This project is licensed under the MIT License. See the license.txt file for details.

---

Enjoy exploring the world of personalized book recommendations! 📖✨
