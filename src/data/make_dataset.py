import pathlib
import yaml
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    X_train, X_test, y_train, y_test= train_test_split(df['stats_features'].to_list(), df['class'].to_list(), test_size=test_split, stratify=df['class'].to_list(), random_state=seed)
    X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=test_split-0.05, stratify=y_train, random_state=seed)
    return X_train, y_train, X_test, y_test, X_val, y_val

def format_data(X_train, y_train, X_test, y_test, X_val, y_val):
    for i in range(len(X_train)):
        X_train[i]=X_train[i].split(',')
        for j in range(len(X_train[i])):
            X_train[i][j]=float(X_train[i][j].strip("[]"))

    for i in range(len(X_test)):
        X_test[i]=X_test[i].split(',')
        for j in range(len(X_test[i])):
            X_test[i][j]=float(X_test[i][j].strip("[]"))

    for i in range(len(X_val)):
        X_val[i]=X_val[i].split(',')
        for j in range(len(X_val[i])):
            X_val[i][j]=float(X_val[i][j].strip("[]"))
    
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    X_val=np.array(X_val)
    y_val=np.array(y_val)

def save_data(X_train, y_train, X_test, y_test, X_val, y_val, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # train.to_csv(output_path + '/train.csv', index=False)
    # test.to_csv(output_path + '/test.csv', index=False)
    # val.to_csv(output_path + '/val.csv', index=False)
    np.save(file=output_path + '/X_train.npy', arr=X_train)
    np.save(file=output_path + '/y_train.npy', arr=y_train)
    np.save(file=output_path + '/X_test.npy', arr=X_test)
    np.save(file=output_path + '/y_test.npy', arr=y_test)
    np.save(file=output_path + '/X_val.npy', arr=X_val)
    np.save(file=output_path + '/y_val.npy', arr=y_val)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    #input_file = sys.argv[1]
    input_file='/data/raw/final_features.csv'
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'
    
    data = load_data(data_path)
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(data, params['test_split'], params['seed'])
    format_data(X_train, y_train, X_test, y_test, X_val, y_val)
    save_data(X_train, y_train, X_test, y_test, X_val, y_val, output_path)

if __name__ == "__main__":
    main()