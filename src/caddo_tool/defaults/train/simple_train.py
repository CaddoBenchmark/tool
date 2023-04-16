from caddo_tool.modules.attributes import Attributes


class SimpleTrain:

    def run(self, attributes):
        attributes[Attributes.MODEL].fit(attributes[Attributes.X], attributes[Attributes.Y])
        return attributes