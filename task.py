import time
import torch
import torch.nn as nn
from model import LeNetModel
from sklearn.metrics import confusion_matrix
from data_loader import load
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # normal ditribution
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            fmt = '.3f'
        else:
            fmt = '0'
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_performance_rates(cm):
    """
    This function caclulates TPR, TNR,  PPV, NPV, FPR, FNR, FDR, and ACC. Then it plots
    TPR, FPR and Accuracy per class.

    :param cm: confusion matrix
    :return:
    """

    false_positive = cm.sum(axis=0) - np.diag(cm)
    false_negative = cm.sum(axis=1) - np.diag(cm)
    true_positive = np.diag(cm)
    true_negative = cm.sum() - (false_positive + false_negative + true_positive)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = true_positive / (true_positive + false_negative)
    print("TPR: " + str(["%.3f" % member for member in TPR]))
    # Specificity or true negative rate
    TNR = true_negative / (true_negative + false_positive)
    print("TNR:" + str(["%.3f" % member for member in TNR]))
    # Precision or positive predictive value
    PPV = true_positive / (true_positive + false_positive)
    print("PPV: " + str(["%.3f" % member for member in PPV]))
    # Negative predictive value
    NPV = true_negative / (true_negative + false_negative)
    print("NPV: " + str(["%.3f" % member for member in NPV]))
    # Fall out or false positive rate
    FPR = false_positive / (false_positive + true_negative)
    print("FPR: " + str(["%.3f" % member for member in FPR]))
    # False negative rate
    FNR = false_negative / (true_positive + false_negative)
    print("FNR: " + str(["%.3f" % member for member in FNR]))

    # False discovery rate
    FDR = false_positive / (true_positive + false_positive)
    print("FDR: " + str(["%.3f" % member for member in FDR]))
    # Overall accuracy
    ACC = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    #printed string for clarity
    print("ACC: " + str(["%.3f" % member for member in ACC]))

    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    f, axarr = plt.subplots(3, 1, sharex='col', sharey='row')
    axarr[0].bar(index, TPR)
    axarr[0].set_title('TPR', color='blue')
    axarr[1].bar(index, FPR, color='red')
    axarr[1].set_title('FPR', color='red')
    axarr[2].bar(index, ACC, color='green')
    axarr[2].set_title('ACC', color='green')

    for ax in axarr.flat:
        ax.set(xlabel='classes', ylabel='rate')
        ax.set_xticks(index)
        ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    for ax in axarr.flat:
        ax.label_outer()
    plt.tight_layout()
    plt.show()

def model_train(net, dataset):

    """
    This function trains the model 
    :param dataset: contains the data that was selected for training
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(30):
        print("Epoch: {}".format(epoch + 1))
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataset, 0):

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')
    #save the parameters of the model
    torch.save(net.state_dict(), './saved_model.pth')



def model_test(net, dataset):
    """
    :param net: The model that we plan to test its accuracy
    :param dataset: contains the test data
    :return: 
    """
    total = 0
    correct = 0
    start_time = time.time()
    time.time()
    #initialize confiution matrix
    cm = np.zeros((10, 10))
    for i, (images, labels) in enumerate(dataset):
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        #labels should be sent to the confusion matrix function
        cm += confusion_matrix(y_true=labels, y_pred=predicted, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    end_time = time.time()
    execution_time = end_time - start_time
    print("execution time on is: {}".format(execution_time))
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %f %%' % (accuracy))
    print("Confusion Matrix for the Model:\n" + np.array_str(cm, precision=2, suppress_small=True))
    plt.figure()
    plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], normalize=True,
                          title='Confusion matrix, with normalization')
    plt.show()
    plot_performance_rates(cm)
    # maximum_false_positive_rate(cm)

    print('Testing is Done!')
    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="classifies the MNIST dataset for jumio task.")
    parser.add_argument("--train", help="train model", default='no', type=str, required=True)
    parser.add_argument("--test", help="test model", default='yes',type=str, required=True)

    args = parser.parse_args()
    print('data is loading ....')
    train_dataset, test_dataset = load()
    print('Data Loading is Done!')
    net = LeNetModel()

    if args.train == 'yes':
        model_train(net=net, dataset=train_dataset)
    if args.test == 'yes':
        net.load_state_dict(torch.load('./saved_model.pth'))
        model_test(net=net, dataset=test_dataset)






