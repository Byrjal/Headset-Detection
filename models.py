from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_models():
    """
    Returns a dictionary of the used machine learning classification models.
    """
    return {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    }