import joblib
import pathlib
import numpy as np

def load_data(data_path):
    X = np.load(data_path)
    return X

def predict(X,model_path):
    model=joblib.load(model_path)
    prediction=model.predict(X)

    return prediction

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file='/data/processed/X_val.npy'
    data_path = home_dir.as_posix() + input_file
    model_path = home_dir.as_posix() + '/models/pipe_model.joblib'
    X=load_data(data_path)
    prediction=predict(X[0].reshape(1,-1),model_path)
    print(prediction)

if __name__ == "__main__":
    main()