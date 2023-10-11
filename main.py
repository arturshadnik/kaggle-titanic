import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from models import LogisticRegression, DNN
from process_data import DataProcessor

def do_data_prep():
    processor = DataProcessor("C://Users//artur//OneDrive - sfu.ca//Desktop//Programming//ML//kaggle-titanic//raw_data//train.csv")
    processor.load_data_from_csv()
    # processor.preview_data(5)

    mean_age = processor.get_mean_of_numerical_column('Age').__round__(2)
    processor.fill_missing_values(["Age"], [mean_age])

    sex_mapping = {"male": 0, "female": 1}
    processor.str_to_label("Sex", sex_mapping)

    unique_embarked = processor.data['Embarked'].unique().tolist()
    embarked_mapping = {}
    for i, key in enumerate(unique_embarked):
        embarked_mapping.update({key: i+1})
    processor.str_to_label("Embarked", embarked_mapping)

    # datatypes = processor.get_all_datatypes()
    numeric_columns = ["SibSp", "Parch", "Fare", "Age", "Pclass", "Embarked"]
    processor.normalize_numeric_columns(numeric_columns)
    columns_to_drop = ["Name", "Ticket", "Cabin", "PassengerId"]
    processor.drop_columns(columns_to_drop)

    # processor.preview_data()

    processor.split_test_train_data("Survived", 0.1)
    X_train, X_dev, Y_train, Y_dev = processor.X_train, processor.X_test, processor.Y_train, processor.Y_test

    return X_train, X_dev, Y_train, Y_dev

def do_model_training(model_type: str, X_train, X_val, Y_train, Y_val, epochs, batch_size):
    input_size = X_train.shape[1]
    if model_type == "logistic_regression":
        model = LogisticRegression("adam", "binary_crossentropy", input_size)
    if model_type == "deep":
        model = DNN("adam", "sparse_categorical_crossentropy", input_size, metrics=["accuracy", "binary_accuracy"])
    
    model.train_model(X_train, Y_train, X_val, Y_val, epochs, batch_size)

    return model

def do_model_inference():
    pass

def main():
    epochs = 500
    batch_size = 256
    filename = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    X_train, X_val, Y_train, Y_val = do_data_prep()

    # model_logistic = do_model_training("logistic_regression", X_train, X_val, Y_train, Y_val, epochs, batch_size)
    model_dnn = do_model_training("deep", X_train, X_val, Y_train, Y_val, epochs, batch_size)
    
    model_dnn.plot_results()
    # print(f"Saving {filename}")
    # model_logistic.save(filename + "_logistic")
    # model_dnn.save(filename + "_dnn")

if __name__ == "__main__":
    main()