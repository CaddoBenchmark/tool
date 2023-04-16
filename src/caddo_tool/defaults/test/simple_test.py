from caddo_tool.modules.attributes import Attributes


class SimpleTest:

    def run(self, attributes):
        attributes[Attributes.Y] = attributes[Attributes.MODEL].predict(attributes[Attributes.X])
        return attributes
