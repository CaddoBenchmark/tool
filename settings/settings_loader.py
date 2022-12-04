import yaml
from yaml import SafeLoader

from settings.settings import Settings


class SettingsLoader:
    def __init__(self, settings_path=''):
        self.settings_path = settings_path

    def load(self):
        print("LOADING SETTINGS:")
        self.load_settings_object()
        print()

    def read_settings_file(self):
        with open(self.settings_path) as f:
            data = yaml.load(f, Loader=SafeLoader)
            return data

    def load_settings_object(self):
        settings_file = self.read_settings_file()
        print("Settings:")
        print(yaml.dump(settings_file, default_flow_style=False))
        Settings.input_data_file = settings_file['data']['path']
        Settings.data_loader_module_path = settings_file['modules']['data_loader']
        Settings.data_preprocessor_module_path = settings_file['modules']['data_preprocessor']
        Settings.net_trainer_module_path = settings_file['modules']['net_trainer']
        Settings.net_tester_module_path = settings_file['modules']['net_tester']
        Settings.net_evaluator_module_path = settings_file['modules']['net_evaluator']
