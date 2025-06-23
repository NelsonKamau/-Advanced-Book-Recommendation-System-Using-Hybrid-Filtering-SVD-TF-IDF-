# 📚 Advanced Book Recommendation System Using Hybrid Filtering (SVD + TF-IDF)

This project implements a hybrid recommendation system using the Book-Crossing dataset. It combines collaborative filtering (SVD) with content-based filtering (TF-IDF + k-NN) to generate accurate and personalized book suggestions.

---

## 📌 Features

* ✅ Data cleaning and preprocessing (Users, Books, Ratings)
* 📊 Exploratory data analysis (EDA) with visualizations
* 🧠 Collaborative filtering using Surprise SVD
* 💾 Content-based filtering using TF-IDF and cosine similarity
* 🔀 Hybrid recommendation logic combining both methods
* 📈 Model evaluation using cross-validation (RMSE, MAE)

---

## 🗂️ Dataset

The [Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) consists of:

* `Users.csv`: user demographics (User-ID, Location, Age)
* `Books.csv`: book metadata (ISBN, Title, Author, Year, Publisher)
* `Ratings.csv`: explicit and implicit book ratings (1–10, 0)

---

## 🧰 Technologies Used

* Python (Pandas, NumPy, Matplotlib, Scikit-learn)
* Surprise (SVD model)
* NLP: TfidfVectorizer
* NearestNeighbors (Cosine similarity)

---

## 🚀 How to Run

1. Clone the repo

   ```bash
   git clone https://github.com/your-username/book-recommendation-hybrid.git
   cd book-recommendation-hybrid
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook
   Open `Books_rec_Enhanced.ipynb` in Jupyter or Colab.

---

## 📌 Example Usage

```python
get_recommendations("Harry Potter and the Sorcerer's Stone")

hybrid_recommendation(
    user_id=276729,
    title="Harry Potter and the Sorcerer's Stone",
    svd_model=svd,
    nn_model=nn_model,
    tfidf_matrix=tfidf_matrix,
    books_df=books
)
```

---

## 📊 Sample Output

Top 5 hybrid recommendations for user 276729:

| Book Title                              | Author              |
| --------------------------------------- | ------------------- |
| Harry Potter and the Chamber of Secrets | J.K. Rowling        |
| Fantastic Beasts and Where to Find Them | J.K. Rowling        |
| The Hobbit                              | J.R.R. Tolkien      |
| Eragon                                  | Christopher Paolini |
| Percy Jackson and the Olympians         | Rick Riordan        |

---

## 📘 License

This project is licensed under the MIT License.
Feel free to use and modify for academic or personal purposes.

---

## 💡 Credits

Built by \Nelson Kamau using the Book-Crossing dataset.
Inspired by hybrid recommender systems used by Amazon and Netflix.
