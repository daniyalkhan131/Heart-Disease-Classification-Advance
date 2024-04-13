from imblearn.over_sampling import SMOTE
import pathlib
import numpy as np
import yaml
import sys

def load_data(data_path):
    X_train, y_train = np.load(data_path+'/X_train.npy'), np.load(data_path+'/y_train.npy')
    return X_train, y_train

def handle_imbalance(X_train, y_train, strategy):
    smote = SMOTE(sampling_strategy=strategy)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    return X_sm, y_sm

def save_data(X_train, y_train, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    np.save(file=output_path + '/X_train.npy', arr=X_train)
    np.save(file=output_path + '/y_train.npy', arr=y_train)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["build_features"]

    # input_file='/data/processed'
    input_file=sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    X_train, y_train= load_data(data_path)
    X_train, y_train=handle_imbalance(X_train, y_train, params['sampling_strategy'])

    output_path = home_dir.as_posix() + '/data/interim'
    save_data(X_train,y_train, output_path)


if __name__ == "__main__":
    main()