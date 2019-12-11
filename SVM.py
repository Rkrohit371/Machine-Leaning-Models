

import pandas as pd
import numpy as np




import matplotlib.pyplot as plt
import seaborn as ans



get_ipython().run_line_magic('matplotlib', 'inline')




from sklearn.datasets import load_breast_cancer





cancer = load_breast_cancer()




cancer.keys()



df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])




from sklearn.model_selection import train_test_split



X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)



from sklearn.svm import SVC




model = SVC()





model.fit(X_train,y_train)




from sklearn.metrics import classification_report,confusion_matrix




import sklearn
from sklearn.model_selection import learning_curve, GridSearchCV





param_grid = {'C':[0.01,0.1,1,10,100],'gamma':[1,0.1,0.01,0.001,0.0001]}




grid = GridSearchCV(SVC(),param_grid,verbose=3)




grid.fit(X_train,y_train)





grid.best_params_




grid.best_estimator_





grid_predictions = grid.predict(X_test)





print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))






