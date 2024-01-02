import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns 

def plot_confusion_matrix(y_true,y_pred,classes):
    cm = confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot = True, fmt="g",cmap='Blues',xticklabels=classes,yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")  
    plt.show()

def plot_history(history):
    plt.figure(figsize=(10,10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train','test'])
    plt.show()

def print_classification_report(y_true,y_pred,classes):
    report = classification_report(y_true,y_pred,target_names=classes)
    print(report)

def pred_plot(X_test,y_true,y_pred,num_dig=4):
    fig,axes = plt.subplots(1,num_dig,figsize=(10,2))
    for i in range(num_dig):
        axes[i].imshow(X_test[i].reshape(28,28),cmap='gray')
        axes[i].set_title(f"True:{y_true[i]} Pred:{y_pred[i]}")
        axes[i].axis('off')
    plt.show()
    