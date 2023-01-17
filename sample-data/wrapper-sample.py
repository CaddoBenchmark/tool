def preprocess(attributes) -> {}:
    from caddo_tool.modules import Attributes
    """
    Preprocess data in framework
    :param data: data read in previous step
    :return: data in same shape but after preprocessing operations
    """
    attributes[Attributes.X] = attributes[Attributes.X_RAW].apply(lambda x: x + 10)
    attributes[Attributes.Y] = attributes[Attributes.Y_RAW]
    return attributes


def init_model(attributes) -> {}:
    from sklearn.tree import DecisionTreeClassifier
    from caddo_tool.modules import Attributes
    attributes[Attributes.MODEL] = DecisionTreeClassifier()
    return attributes


def train(attributes) -> {}:
    from caddo_tool.modules import Attributes
    """
    Make some training routines on provided data
    :param attributes: data loaded and preprocessed above
    :return: ready to test network
    """
    attributes[Attributes.MODEL].fit(attributes[Attributes.X], attributes[Attributes.Y])
    return attributes


def test(attributes) -> {}:
    from caddo_tool.modules import Attributes
    """
    Make tests on provided X data
    :param data_x: only X data, different from one provided in training step
    :param network: network trained previously
    :return: predictions that will be processed in next step
    """
    attributes[Attributes.Y] = attributes[Attributes.MODEL].predict(attributes[Attributes.X])
    return attributes


def evaluate(attributes):
    from caddo_tool.modules import Attributes
    print(attributes[Attributes.Y])
    print(attributes[Attributes.Y_TRUE])
