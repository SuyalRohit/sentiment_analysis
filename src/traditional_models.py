from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(max_iter=1000),
        "MultinomialNB": MultinomialNB()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")
    
    return models