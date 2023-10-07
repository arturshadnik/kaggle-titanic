import pandas as pd
from typing import List, Any
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

from basic_model import basic_model

class DataProcessor:

    def __init__(self, path_to_data: str) -> None:
        self.path_to_data = path_to_data
        print(f"Initialized data processor for {os.path.basename(self.path_to_data)}")

    def load_data_from_csv(self):
        try:
            self.data = pd.read_csv(self.path_to_data)
            print("Successfully loaded data")
        except Exception as e:
            print(e)
            self.data = pd.DataFrame()

    def preview_data(self, number_of_rows=5):
        print(self.data.info())
        print(self.data.head(number_of_rows))
    
    def get_mean_of_numerical_column(self, column):
        try:
            return self.data[column].mean()
        except Exception as e:
            print(e)
            return None

    def fill_missing_values(self, columns_to_fill: List[str], default_values: List[Any]):
        for i, column in enumerate(columns_to_fill):
            try:
                self.data[column].fillna(default_values[i], inplace=True)
                print(f"Filled missing {column} with {default_values[i]}")
            except Exception as e:
                print(e)

    def drop_columns(self, columns_to_drop: List[Any]):
        for column in columns_to_drop:
            try:
                self.data.drop(column, axis=1, inplace=True)
                print(f"Dropped column {column}")
            except Exception as e:
                print(e)                

    def drop_rows_with_missing_values(self, columns_to_drop: List[Any]): 
        for column in columns_to_drop:
            try:
                self.data.dropna(subset=[column], inplace=True)
                print(f"Removed missing rows for {column}")
            except Exception as e:
                print(e)

    def get_all_datatypes(self):
        return self.data.dtypes
    
    def normalize_numeric_columns(self, numeric_columns):
        scaler = StandardScaler()
        self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])

    def split_test_train_data(self, target_column_name, test_size=None):
        x_temp = self.data.drop(target_column_name, axis=1)
        y_temp = self.data[target_column_name]

        if test_size is not None:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x_temp, y_temp, test_size=test_size, random_state=42)
        else:
             self.X_train, self.X_test, self.Y_train, self.Y_test = x_temp, None, y_temp, None
        
def train_model(X_train, Y_train):
    basic_model(X_train, Y_train, 100)


## ! ! ! TODO deal with str more elegantly, automate normalizing numeric columns
if __name__ == "__main__":
    processor = DataProcessor(os.getenv("TRAIN_DATA_PATH"))
    processor.load_data_from_csv()
    processor.preview_data(5)

    mean_age = processor.get_mean_of_numerical_column('Age').__round__(2)
    processor.fill_missing_values(["Age"], [mean_age])

    datatypes = processor.get_all_datatypes()
    numeric_columns = ["Pclass","SibSp", "Parch", "Fare"]
    processor.normalize_numeric_columns(numeric_columns)
    columns_to_drop = ["Name", "Sex", "Ticket", "Cabin", "Embarked", "PassengerId"]
    processor.drop_columns(columns_to_drop)

    processor.preview_data()

    processor.split_test_train_data("Survived")
    X_train, X_test, Y_train, Y_test = processor.X_train, processor.X_test, processor.Y_train, processor.Y_test
    print(X_train.shape)
    # print(Y_test.shape)

    model = train_model(X_train, Y_train)
    print(model)