from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.visualization import confusion_matrix_visualization

import sys
sys.path.append('..')
from utils.utils import accuracy_roughly

# Gradient Boosting classifier
def Grad_Boost(X_train, X_test, y_train, y_test,max_depth = 3):
    GB = GradientBoostingClassifier(max_depth=max_depth,random_state=42, n_estimators=256)
    GB.fit(X_train, y_train)
    y_pred = GB.predict(X_test)

    # accuracy
    print("Gradient Boosting Classifier")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("accuracy(roughly):", accuracy_roughly(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))
    confusion_matrix_visualization(clf=GB, X_test= X_test, y_test=y_test)
    
    acc = accuracy_score(y_pred, y_test)*100
    acc_rough = accuracy_roughly(y_pred, y_test)*100

    return  acc, acc_rough