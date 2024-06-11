
# Fake News Detection Using Machine Learning

## Project Overview

This project aims to distinguish between real and fake news articles using machine learning techniques. By leveraging Natural Language Processing (NLP) and a Naive Bayes classifier, the model can accurately predict the authenticity of a given news article.

## Features

- **Data Source**: Combined datasets of true and fake news articles.
- **Text Preprocessing**: Tokenization, stop-word removal, and stemming to clean and prepare the text data.
- **Feature Extraction**: TF-IDF vectorization to convert text data into numerical features.
- **Model Training**: Multinomial Naive Bayes classifier for training and prediction.
- **User Interaction**: Allows users to input a news article and get a real-time prediction of its authenticity.

## Usage

1. **Preprocess the Data**: The text data is tokenized, cleaned of stop words, and stemmed.
2. **Feature Extraction**: The cleaned text is transformed into TF-IDF vectors.
3. **Model Training**: The model is trained using the preprocessed data.
4. **Prediction**: Users can input a news article to get a prediction.

## Example

### Preprocess a Fake News Article

```python
news_idk = """Neutral-Atom Breakthrough Surpasses Quantum Error-Correcting Thresholds, A Critical Step to Achieving Quantum Viability in Commercial Applications 
BOSTON, MASSACHUSETTS, October 12, 2023 – QuEra Computing, the leader in neutral-atom quantum computers, today announced that a team of researchers from Harvard, MIT and QuEra successfully demonstrated two-qubit entangling gates with an unprecedented 99.5% fidelity on 60 neutral atom qubits in parallel. The quantum breakthrough is the result of an extensive test conducted by Harvard University’s Department of Physics and John A. Paulson School of Engineering and Applied Sciences, QuEra, and MIT’s Department of Physics and Research Laboratory of Electronics. The breakthrough was first reported in ArXiv, and the full research paper can be found here.

Performing entangling quantum operations with low error rates in a scalable fashion is a central element of useful quantum information processing. Neutral atom arrays have recently emerged as a promising quantum computing platform, featuring coherent control over hundreds of qubits and any-to-any gate connectivity in a flexible, dynamically reconfigurable architecture. Fidelities of above 99% (or below 1% error rates) are required to surpass quantum error-correcting thresholds, and previously, 97.5% was the highest fidelity achieved within this configuration. By enabling high-fidelity operation in a scalable, highly connected system, MIT, Harvard, and QuEra have laid the groundwork for large-scale implementation of quantum algorithms, error-corrected circuits, and digital simulations"""

news_idk_processed = process(news_idk)
news_idk_processed = " ".join(news_idk_processed)

# Transform the preprocessed string into a TF-IDF vector
news_idk_tfidf = vectorizer.transform([news_idk_processed]).toarray()

# Predict the class of the news article
prediction = nb.predict(news_idk_tfidf)
print(f"The prediction for news_idk is: {'Real' if prediction[0] == 1 else 'Fake'}")
```

## Data Sources

- Fake news article from [FactCheck.org](https://www.factcheck.org/)
- Real news article from [QuEra Computing](https://www.quera.com/)

