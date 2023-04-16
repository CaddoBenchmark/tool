from pathlib import Path

from caddo_tool.modules.attributes import Attributes
import matplotlib.pyplot as plt


class SummarizeAll:

    def run(self, attributes, figures_destination):
        Path(figures_destination).mkdir(parents=True, exist_ok=True)
        acc = attributes[Attributes.STORE]['acc']
        rec = attributes[Attributes.STORE]['rec']
        blc_acc = attributes[Attributes.STORE]['blc_acc']
        f1 = attributes[Attributes.STORE]['f1']
        l_loss = attributes[Attributes.STORE]['l_loss']
        prec =  attributes[Attributes.STORE]['prec']
        jc = attributes[Attributes.STORE]['jc']
        runs = [x for x in range(len(acc))]
        # Accuracy
        plt.plot(runs, acc)
        plt.xlabel("Runs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/accuracy.png")
        plt.clf()
        # Recall
        plt.plot(runs, rec)
        plt.xlabel("Runs")
        plt.ylabel("Recall")
        plt.title("Recall vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/recall.png")
        plt.clf()
        # Balanced Recall
        plt.plot(runs, blc_acc)
        plt.xlabel("Runs")
        plt.ylabel("Balanced Recall")
        plt.title("Balanced Recall vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/balanced_recall.png")
        plt.clf()
        # F1 Score
        plt.plot(runs, f1)
        plt.xlabel("Runs")
        plt.ylabel("F1")
        plt.title("F1 vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/f1.png")
        plt.clf()
        # Log Loss
        plt.plot(runs, l_loss)
        plt.xlabel("Runs")
        plt.ylabel("Log Loss")
        plt.title("Log Loss vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/log_loss.png")
        plt.clf()
        # Precision
        plt.plot(runs, prec)
        plt.xlabel("Runs")
        plt.ylabel("Precision")
        plt.title("Precision vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/precision.png")
        plt.clf()
        # Jaccard
        plt.plot(runs, jc)
        plt.xlabel("Runs")
        plt.ylabel("Jaccard")
        plt.title("Jaccard vs Runs")
        plt.grid(True)
        plt.savefig(figures_destination + "/jaccard.png")
