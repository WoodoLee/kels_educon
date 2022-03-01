from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
sys.path.append('..')
from utils.utils import accuracy_roughly
from utils.visualization import confusion_matrix_visualization

# Random Forest Classifier
def RandomForest(X_train, X_test, y_train, y_test):
    random_forest = RandomForestClassifier(criterion='entropy', max_depth=32, bootstrap=True, random_state=42, n_estimators=256, min_samples_split=2)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)

    # accuracy
    print("Random Forest Classifier")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("accuracy(roughly):", accuracy_roughly(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))
    print()
    print("confusion matrix visualization")
    confusion_matrix_visualization(clf=random_forest, X_test= X_test, y_test=y_test)

    
    acc = accuracy_score(y_pred, y_test)*100
    acc_rough = accuracy_roughly(y_pred, y_test)*100

    return  acc, acc_rough
