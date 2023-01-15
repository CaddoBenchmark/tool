# preprocess data in framework
def preprocess(attributes) -> {}:
    from modules.attributes import Attributes
    """
    Preprocess data in framework
    :param data: data read in previous step
    :return: data in same shape but after preprocessing operations
    """
    attributes[Attributes.X] = attributes[Attributes.X_RAW]
    attributes[Attributes.Y] = attributes[Attributes.Y_RAW]
    return attributes


def init_model(attributes) -> {}:
    from sklearn.tree import DecisionTreeClassifier
    from modules.attributes import Attributes
    attributes[Attributes.MODEL] = DecisionTreeClassifier()
    return attributes


def train(attributes) -> {}:
    from modules.attributes import Attributes
    """
    Make some training routines on provided data
    :param attributes: data loaded and preprocessed above
    :return: ready to test network
    """
    print("Training...")
    attributes[Attributes.MODEL].fit(attributes[Attributes.X], attributes[Attributes.Y])


def test(attributes) -> {}:
    from modules.attributes import Attributes
    """
    Make tests on provided X data
    :param data_x: only X data, different from one provided in training step
    :param network: network trained previously
    :return: predictions that will be processed in next step
    """
    print("Testing...")
    attributes[Attributes.MODEL].predict(attributes[Attributes.Y])
    return {}


def evaluate(data, predictions):
    print("Evaluating...")
