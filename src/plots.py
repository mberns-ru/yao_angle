''' All the plotting functions are defined here. '''

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
mpl.use('Agg') # This is needed to save the plots in a non-interactive mode


def plot_list(l, png_fn):
    ''' plot a list into a png file.'''
    plt.figure(figsize=(24, 10))
    sns.set(style='ticks')
    sns.set_context('poster')  # Increase line width
    plt.plot(l)
    plt.savefig(f'{png_fn}.png')
    plt.close()


def plot_cross_validation_results(avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies, pdf_file_name):
    ''' Plot how well cross validation went.'''
    epochs = range(1, len(avg_train_losses) + 1)
    plt.figure(figsize=(24, 10))
    plt.rcParams['pdf.fonttype'] = 42
    sns.set(style='ticks')
    sns.set_context('poster')  # Increase line width
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_train_losses, 'bo-', label='Average Training loss')
    plt.plot(epochs, avg_val_losses, 'r*-', label='Average Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_train_accuracies, 'bo-',
             label='Average Training accuracy')
    plt.plot(epochs, avg_val_accuracies, 'r*-',
             label='Average Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{pdf_file_name}.pdf',
                transparent=True, format='pdf')
    # plt.show()
    # reset for next plot
    plt.close()


def plot_matrix(matrix, class_names, pdf_fn):
    ''' Ploat a confusion matrix. '''
    precision_matrix = pd.DataFrame(
        matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(precision_matrix, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Precision Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{pdf_fn}_confusion.pdf',
                transparent=True, format='pdf')
    plt.close()
    # plt.show()
