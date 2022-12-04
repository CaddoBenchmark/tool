def load() -> {}:
    """
    Loads data into framework
    :return: dictionary with loaded data
    """
    data = {'a': 'b'}
    print("Data loading...")
    return data


# preprocess data in framework
def preprocess(data) -> {}:
    """
    Preprocess data in framework
    :param data: data read in previous step
    :return: data in same shape but after preprocessing operations
    """
    print(data)
    print("Preprocessing file...")
    return data


def train(data) -> object:
    """
    Make some training routines on provided data
    :param data: data loaded and preprocessed above
    :return: ready to test network
    """
    print("Training...")
    return None


def test(data_x, network) -> {}:
    """
    Make tests on provided X data
    :param data_x: only X data, different from one provided in training step
    :param network: network trained previously
    :return: predictions that will be processed in next step
    """
    print("Testing...")
    return {}


def evaluate(data, predictions):
    print("Evaluating...")
