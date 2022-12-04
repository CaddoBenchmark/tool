from modules.module_loader import ModuleLoader
from settings.settings_loader import SettingsLoader


class Caddo:
    def __init__(self):
        SettingsLoader('./settings.yaml').load()
        self.data_loader_module = None
        self.data_preprocessor_module = None
        self.net_trainer_module = None
        self.net_tester_module = None
        self.net_evaluator_module = None
        self.load_modules()
        self.run()

    def load_modules(self):
        print("LOADING MODULES:")
        module_loader = ModuleLoader()
        self.data_loader_module = module_loader.load_data_loader()
        self.data_preprocessor_module = module_loader.load_data_preprocessor()
        self.net_trainer_module = module_loader.load_net_trainer()
        self.net_tester_module = module_loader.load_net_tester()
        self.net_evaluator_module = module_loader.load_net_evaluator()
        print()

    def run(self):
        print("RUNNING TESTS")
        network = None
        data = dict()
        data_x = None
        predictions = None
        data = self.data_loader_module.load(data)
        data = self.data_preprocessor_module.preprocess(data)
        network = self.net_trainer_module.train(data)
        predictions = self.net_tester_module.test(data_x, network)
        self.net_evaluator_module.evaluate(data, predictions)


if __name__ == '__main__':
    Caddo()
