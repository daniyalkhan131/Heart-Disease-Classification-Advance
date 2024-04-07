from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import pathlib
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sys

def scaling(X_train):
    scaler=StandardScaler()
    X_train_scale=scaler.fit_transform(X_train)

    return X_train_scale

def dimensionality_reduction(X_train):
    pca=PCA(n_components=7)
    X_train_trf=pca.fit_transform(X_train)

    return X_train_trf


def train_model(X_train, y_train, param_grid, cv):
    # Train your machine learning model
    clf_svm = SVC()
    svm_grid = GridSearchCV(estimator = clf_svm,
                       param_grid = param_grid,
                       cv = cv, #for every combination 5 folds done to get better estimation
                       verbose=3)
    svm_grid.fit(X_train,y_train)

    return svm_grid.best_estimator_

def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')

def model_pipeline(X_train,y_train,param_grid,cv):

    pipe = Pipeline([
        ('scaling',StandardScaler()),
        ('pca',PCA(n_components=7)),
        ('svm',SVC())
    ])

    pipe_grid = GridSearchCV(estimator = pipe,
                       param_grid = param_grid,
                       cv = cv, #for every combination 5 folds done to get better estimation
                       verbose=3)
    pipe_grid.fit(X_train,y_train)

    return pipe_grid.best_estimator_

def save_pipe(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/pipe_model.joblib')

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file = sys.argv[1]
    #input_file='/data/interim'
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    X_train = np.load(data_path + '/X_train.npy')
    y_train = np.load(data_path + '/y_train.npy')
    #X_train=scaling(X_train)
    #X_train=dimensionality_reduction(X_train)
    #trained_model = train_model(X_train, y_train, params['params_grid'], cv=params['cv'])
    trained_model = model_pipeline(X_train, y_train, params['params_grid_pipe'], cv=params['cv'])
    # save_model(trained_model, output_path)
    save_pipe(trained_model, output_path)

    

if __name__ == "__main__":
    main()