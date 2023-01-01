def load() -> {}:
    import pandas as pd
    """
    Loads data into framework
    :return: dictionary with loaded data
    """
    print("Data loading...")
    dataset_df = pd.read_csv("./dataset.csv", sep="$")
    dataset_df = pd.concat([dataset_df[dataset_df['class_value'] == 1],
                            dataset_df[dataset_df['class_value'] == 0].sample(128)], axis=0)
    dataset_df.head(5)
    return {
        "data": dataset_df,
    }


# preprocess data in framework
def preprocess(attributes) -> {}:
    import numpy as np
    """
    Preprocess data in framework
    :param data: data read in previous step
    :return: data in same shape but after preprocessing operations
    """
    print("Preprocessing file...")
    print(attributes)
    keywords = ['"', "sql", "statement", "select", "insert", "delete", "update", "drop", "execute"]
    result = []
    for x in attributes["data"]['contents'].astype(str):
        result.append(np.array([1 if keyword in x else 0 for keyword in keywords]))
    attributes["X"] = np.array(result).reshape(-1, len(keywords))
    attributes["Y"] = attributes["data"]['class_value']
    return attributes


def init_model(attributes) -> {}:
    from sklearn.tree import DecisionTreeClassifier
    attributes["model"] = DecisionTreeClassifier()
    return attributes


def train(attributes) -> {}:
    """
    Make some training routines on provided data
    :param attributes: data loaded and preprocessed above
    :return: ready to test network
    """
    print("Training...")
    attributes["model"].fit(attributes["X"], attributes["Y"])


def test(attributes) -> {}:
    """
    Make tests on provided X data
    :param data_x: only X data, different from one provided in training step
    :param network: network trained previously
    :return: predictions that will be processed in next step
    """
    print("Testing...")
    attributes["model"].predict(attributes["X"])
    return {}


def evaluate(data, predictions):
    print("Evaluating...")
