import caddo_tool
from caddo_tool.modules.attributes import Attributes


class RewritePreprocess:
    def run(self, attributes):
        attributes[Attributes.X] = attributes[Attributes.X_RAW]
        attributes[Attributes.Y] = attributes[Attributes.Y_RAW]
        return attributes

    def run(self, attributes, y_subset):
        attributes[Attributes.X] = attributes[Attributes.X_RAW]
        attributes[Attributes.Y] = attributes[Attributes.Y_RAW][y_subset]
        return attributes
