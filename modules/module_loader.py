import sys

from settings.settings import Settings


class ModuleLoader:
    def load_data_loader(self):
        module = self.load_module(Settings.data_loader_module_path)
        print("Data Loader loaded!")
        return module

    def load_data_preprocessor(self):
        module = self.load_module(Settings.data_preprocessor_module_path)
        print("Data Preprocessor loaded!")
        return module

    def load_net_trainer(self):
        module = self.load_module(Settings.net_trainer_module_path)
        print("Net Trainer loaded!")
        return module

    def load_net_tester(self):
        module = self.load_module(Settings.net_tester_module_path)
        print("Net Tester loaded!")
        return module

    def load_net_evaluator(self):
        module = self.load_module(Settings.net_evaluator_module_path)
        print("Net Evaluator loaded!")
        return module

    def load_module(self, path_to_module):
        __import__(path_to_module)
        return sys.modules[path_to_module]
