from caddo_tool.modules.attributes import Attributes
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, f1_score, log_loss, precision_score, jaccard_score


class EvaluateAll:

    def run(self, attributes):
        acc = accuracy_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y])
        rec = recall_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y], average='micro')
        blc_acc = balanced_accuracy_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y])
        f1 = f1_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y], average='micro')
        l_loss = log_loss(attributes[Attributes.Y_TRUE], attributes[Attributes.Y])
        prec = precision_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y])
        jc = jaccard_score(attributes[Attributes.Y_TRUE], attributes[Attributes.Y])

        if 'acc' not in attributes[Attributes.STORE]:
            attributes[Attributes.STORE]['acc'] = [acc]
            attributes[Attributes.STORE]['rec'] = [rec]
            attributes[Attributes.STORE]['blc_acc'] = [blc_acc]
            attributes[Attributes.STORE]['f1'] = [f1]
            attributes[Attributes.STORE]['l_loss'] = [l_loss]
            attributes[Attributes.STORE]['prec'] = [prec]
            attributes[Attributes.STORE]['jc'] = [jc]
        else:
            attributes[Attributes.STORE]['acc'].append(acc)
            attributes[Attributes.STORE]['rec'].append(rec)
            attributes[Attributes.STORE]['blc_acc'].append(blc_acc)
            attributes[Attributes.STORE]['f1'].append(f1)
            attributes[Attributes.STORE]['l_loss'].append(l_loss)
            attributes[Attributes.STORE]['prec'].append(prec)
            attributes[Attributes.STORE]['jc'].append(jc)
        return attributes
