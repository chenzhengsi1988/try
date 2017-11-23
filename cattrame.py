import pandas as pd

TRAIN_URL = "data/iris_training.csv"
TEST_URL = "data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
CSV_COLUMN_NAMES_1 = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species', 'Color']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

color_map = {0:'red', 1:'blue', 2:'white'}
add_column = 'Color'

def load_data():
    # ""Returns the iris dataset as (train_x, train_y), (test_x, test_y)
    train_path = TRAIN_URL
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)

   train[add_column] = train.apply(lambda row: color_map[row.Species], axis=1)
    train.to_csv(‘data/iris_training_color.csv’)
    #print(pd.read_csv(‘data/iris_training_color.csv’, names=CSV_COLUMN_NAMES_1, header=0))

   test_path = TEST_URL
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test[add_column] = test.apply(lambda row: color_map[row.Species], axis=1)
    test.to_csv(‘data/iris_test_color.csv’)
    print(pd.read_csv(‘data/iris_test_color.csv’, names=CSV_COLUMN_NAMES_1, header=0))

if __name__ == ‘__main__‘:
    load_data()