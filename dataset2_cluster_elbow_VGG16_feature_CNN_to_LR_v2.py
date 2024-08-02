import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.metrics import silhouette_score

plt.rcParams.update({'font.size': 14})

# Function to load *.npy files from a folder
def load_and_resize_images(folder, categories):
    X = []
    y = []
    for category in categories:
        class_folder = os.path.join(folder, category)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith('.npy'):
                    img_path = os.path.join(class_folder, filename)
                    feature = np.load(img_path)  
                    X.append(feature)
                    y.append(category)
    return np.array(X), np.array(y)

# Function to normalize the data using Gaussian normalization
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

# Function to perform K-means clustering
def perform_kmeans(X_train_normalized, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train_normalized)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

# Function to determine optimal number of clusters using the Elbow Method
def elbow_method(result_out, X, max_clusters=10, filename ='elbow_method'):
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    # Find the elbow point using the "knee" method
    deltas = np.diff(inertia)
    second_deltas = np.diff(deltas)
    elbow_point = np.argmax(second_deltas) + 2  # +2 because diff reduces the array length by 1 each time

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.axvline(x=elbow_point, color='r', linestyle='--')
    # plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    # plt.show()
    plt.savefig(os.path.join(result_out, f'{filename}.png'))
    
    return elbow_point

# Function to create the Keras model
def create_keras_model(input_shape, num_categories, plot_dir):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Plot and save the model architecture
    plot_path = os.path.join(plot_dir, 'model_architecture.png')
    plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
    print(f"Model architecture plot saved to {plot_path}")

    return model


def plot_roc_curve(y_true, y_pred_probs, class_labels, result_out, reportName):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', len(class_labels))  # Get a color map with enough colors
    line_styles = ['-', '--', '-.', ':']  # Different line styles
    markers = ['o', 'x', 'v', '^', 'd', '*']  # Different markers
    
    if len(class_labels) == 2:  # Binary classification case
        y_true_binary = (y_true == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, color=colors(1), label=f'{class_labels[1]} (AUC = {roc_auc:.4f})')
        # plt.plot(fpr, tpr, color=colors(1), linestyle=line_styles[0], marker='o', label=f'{class_labels[1]} (AUC = {roc_auc:.4f})')
        plt.plot(fpr, tpr, color=colors(1), linestyle=line_styles[0], marker=markers[0], label=f'{class_labels[1]} (AUC = {roc_auc:.4f})')
        
        # Plotting for the other class (0)
        y_true_binary = (y_true == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, color=colors(0), label=f'{class_labels[0]} (AUC = {roc_auc:.4f})')
        # plt.plot(fpr, tpr, color=colors(0), linestyle=line_styles[1], marker='x', label=f'{class_labels[0]} (AUC = {roc_auc:.4f})')
        plt.plot(fpr, tpr, color=colors(0), linestyle=line_styles[1], marker=markers[1], label=f'{class_labels[0]} (AUC = {roc_auc:.4f})')
    else:  # Multi-class case
        for i, class_name in enumerate(class_labels):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors(i), label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    # plt.title('ROC Curves')
    plt.savefig(os.path.join(result_out, f'{reportName}_roc_curves.png'))
    print(f"ROC curve saved to {os.path.join(result_out, f'{reportName}_roc_curves.png')}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_labels, result_out, reportName):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title(f'Confusion Matrix for {reportName}')
    # Save the confusion matrix plot
    plt.savefig(os.path.join(result_out, f'{reportName}_confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(result_out, f'{reportName}_confusion_matrix.png')}")
    plt.close()
    

    
# Function to plot precision-recall curve
def plot_precision_recall(y_true, y_pred_probs, class_labels, result_out, reportName):
    y_true = np.array(y_true)  # Ensure y_true is a numpy array
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_labels):
        y_true_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, i])
        plt.plot(recall, precision, label=f'{class_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(result_out, f'{reportName}_precision_recall.png'))
    print(f"Precision-recall curve saved to {os.path.join(result_out, f'{reportName}_precision_recall.png')}")
    plt.close()

def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    return accuracy, error_rate, precision, recall, f1, sensitivity, specificity


# Function to plot training history
def plot_history(history, test_acc, result_folder, filename="training_history"):
    plt.figure(figsize=(12, 4))

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    if len(test_acc) == len(history['accuracy']):
        plt.plot(test_acc, linestyle='--', label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(result_folder, f"{filename}.png"))
    plt.close()



# Function to calculate and save metrics
def calculate_metrics(y_true, y_pred, y_pred_probs, class_labels, result_out, reportName, epoch, cluster_id, model_name, train_time):
    accuracy, error_rate, precision, recall, f1, sensitivity, specificity = calculate_classification_metrics(y_true, y_pred)
    
    plot_roc_curve(y_true, y_pred_probs, class_labels, result_out, reportName)
    plot_confusion_matrix(y_true, y_pred, class_labels, result_out, reportName)
    plot_precision_recall(y_true, y_pred_probs, class_labels, result_out, reportName)

    metrics_dict = {
        'epoch': epoch,
        'cluster_id': cluster_id,
        'model_name': model_name,
        'Accuracy': accuracy,
        'Error Rate': error_rate,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Train Time': train_time
    }
    
    return metrics_dict

# Function to create the stacked dataset
def stacked_dataset(members, inputX):
    if len(members) == 0:
        raise ValueError("No models found to create the stacked dataset.")
    stackX = None
    for model in members:
        yhat = model.predict(inputX, verbose=0)
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

# Function to load all models
def load_all_models(epoch, num_clusters, result_folder):
    all_models = list()
    result_out = os.path.join(result_folder, f'epoch_{epoch}_num_clusters_{num_clusters}')
    for i in range(num_clusters):
        filename = os.path.join(result_out, f'cluster_{i}.h5')
        if not os.path.exists(filename):
            print(f'Model file {filename} not found.')
            continue
        model = load_model(filename)
        all_models.append(model)
        print(f'>loaded {filename}')
    return all_models

# Function to evaluate the stacked model
def evaluate_stack_model(epoch, clusters, model_name, stacked_test_pred, stacked_test_probabilities, test_labels, result_out, reportName, class_labels, train_time):
    y_true = np.array(test_labels)  # Ensure test_labels is a numpy array
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)  # Fit the label encoder on class labels

    # Encode test labels
    encoded_test_labels = label_encoder.transform(y_true)
    encoded_stacked_test_pred = label_encoder.transform(stacked_test_pred)  # Ensure predictions are encoded as well

    # Calculate metrics
    accuracy, error_rate, precision, recall, f1, sensitivity, specificity = calculate_classification_metrics(encoded_test_labels, encoded_stacked_test_pred)
    
    # Plot ROC curves
    plot_roc_curve(encoded_test_labels, stacked_test_probabilities, class_labels, result_out, reportName)

    # Plot confusion matrix
    plot_confusion_matrix(encoded_test_labels, encoded_stacked_test_pred, class_labels, result_out, reportName)

    # Plot precision-recall curve
    plot_precision_recall(encoded_test_labels, stacked_test_probabilities, class_labels, result_out, reportName)

    # Combine all the metrics into a dictionary
    metrics_dict = {
        'epoch': epoch,
        'cluster': clusters,
        'model_name': model_name,
        'Accuracy': accuracy,
        'Error Rate': error_rate,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Train Time': train_time
    }

    return metrics_dict


# Function to train model on each cluster
def train_cluster_models(train_images_normalized, train_labels, val_images_normalized, val_labels, test_images_normalized, test_labels, cluster_labels, epoch, input_shape, num_categories, result_out, num_clusters, categories):
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)  # Fit the label encoder
    cluster_models = []

    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_X = train_images_normalized[cluster_mask]
        cluster_y = train_labels[cluster_mask]

        # Ensure cluster_y is encoded before passing to the model
        cluster_y_encoded = label_encoder.transform(cluster_y)
        cluster_y_categorical = to_categorical(cluster_y_encoded, num_classes=num_categories)

        # Validation split for cluster data
        cluster_X_train, cluster_X_val, cluster_y_train, cluster_y_val = train_test_split(cluster_X, cluster_y_categorical, test_size=0.2, random_state=42)

        model = create_keras_model(input_shape, num_categories, result_out)
        start_time = datetime.now()
        
        # Initialize history dictionary
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        test_acc_per_epoch = []

        for e in range(epoch):
            hist = model.fit(cluster_X_train, cluster_y_train, validation_data=(cluster_X_val, cluster_y_val), epochs=1, batch_size=32, verbose=2)
            for key in hist.history:
                history[key].extend(hist.history[key])
            test_loss, test_acc = model.evaluate(test_images_normalized, to_categorical(label_encoder.transform(test_labels), num_classes=num_categories), verbose=0)
            test_acc_per_epoch.append(test_acc)
        
        end_time = datetime.now()
        time_taken = end_time - start_time

        plot_history(history, test_acc_per_epoch, result_out, f'cluster_{cluster_id}_history')

        model_filename = os.path.join(result_out, f'cluster_{cluster_id}.h5')
        print(f'Saving model to {model_filename}')
        model.save(model_filename)
        cluster_models.append(model)
        
    return cluster_models


# Function to evaluate models and save results
def evaluate_cluster_models(cluster_models, test_images_normalized, test_labels, result_out, class_labels, epoch):
    label_encoder = LabelEncoder()
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_categorical = to_categorical(test_labels_encoded, num_classes=len(class_labels))
    metrics_list = []

    for i, model in enumerate(cluster_models):
        start_time = datetime.now()
        test_loss, test_accuracy = model.evaluate(test_images_normalized, test_labels_categorical, verbose=2)
        end_time = datetime.now()
        time_taken = end_time - start_time

        y_pred_probs = model.predict(test_images_normalized)
        y_pred = np.argmax(y_pred_probs, axis=1)
        metrics = calculate_metrics(test_labels_encoded, y_pred, y_pred_probs, class_labels, result_out, f'cluster_{i}', epoch, i, f'cluster_{i}', time_taken.total_seconds())
        metrics_list.append(metrics)
    
    return metrics_list

# Function to fit stacked model
from sklearn.linear_model import LogisticRegression

# Function to fit stacked model using Logistic Regression
def fit_stacked_model(members, inputX, inputy):
    stackedX = stacked_dataset(members, inputX)
    model = LogisticRegression(max_iter=1000)  # Using Logistic Regression
    model.fit(stackedX, inputy)
    return model

# Function to make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    stackedX = stacked_dataset(members, inputX)
    yhat = model.predict(stackedX)
    return yhat

# Main function remains the same
def main():
    # Set up directories
    num_clusters = 6  # Choose the number of clusters

    directory_work = os.getcwd()
    directory_feature = os.path.join(directory_work, 'Hien_Data_Feature')
    model_name = f'dataset2_cluster_elbow_VGG16_feature_CNN_to_LR_v2'
    result_folder = os.path.join(directory_work, model_name)
    os.makedirs(result_folder, exist_ok=True)

    train_folder = os.path.join(directory_feature, 'train_VGG16_FC_2')
    test_folder = os.path.join(directory_feature, 'test_VGG16_FC_2')

    # Define the categories
    categories = ['contact_dermatitis', 'atopic_dermatitis']
    num_categories = len(categories)

    # Load and resize train and test sets
    train_images, train_labels = load_and_resize_images(train_folder, categories)
    test_images, test_labels = load_and_resize_images(test_folder, categories)

    # Reshape feature data to 1D vectors
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Normalize the feature data
    train_images_normalized, test_images_normalized = normalize_data(train_images, test_images)

    # Split training data into training, validation, and testing sets
    val_images_normalized, test_images_normalized, val_labels, test_labels = train_test_split(test_images_normalized, test_labels, test_size=0.5, random_state=42)

    # Determine the optimal number of clusters using the Elbow Method
    num_clusters = elbow_method(result_folder,train_images_normalized)
    print(f"Optimal number of clusters determined by Elbow Method: {num_clusters}")
    
    # Perform K-means clustering
    cluster_labels, cluster_centers = perform_kmeans(train_images_normalized, num_clusters)
    
    print(f'cluster_labels: {cluster_labels}')
    
    # Train models on each cluster
    input_shape = train_images_normalized.shape[1:]
    
    # epoch_values = [3, 10, 20, 40, 60, 80, 100]
    epoch_values = [3, 10, 20, 40, 60, 80, 100, 120, 150, 120, 150, 180, 200]
    # epoch_values = [2]

    metrics_list = []

    for epoch in epoch_values:
        result_out = os.path.join(result_folder, f'epoch_{epoch}_num_clusters_{num_clusters}')
        os.makedirs(result_out, exist_ok=True)
        
        cluster_models = train_cluster_models(train_images_normalized, train_labels, val_images_normalized, val_labels, test_images_normalized, test_labels, cluster_labels, epoch, input_shape, num_categories, result_out, num_clusters, categories)

        # Evaluate models and save results
        cluster_metrics = evaluate_cluster_models(cluster_models, test_images_normalized, test_labels, result_out, categories, epoch)
        metrics_list.extend(cluster_metrics)

        # Load models for evaluation
        loaded_models = load_all_models(epoch, num_clusters, result_folder)
        if len(loaded_models) == 0:
            print("No models loaded. Exiting.")
            return

        # Fit stacked model on the validation data
        start_time = datetime.now()
        stacked_model = fit_stacked_model(loaded_models, val_images_normalized, val_labels)
        end_time = datetime.now()
        time_taken = end_time - start_time
        
        # Evaluate the stacked model on the test data
        stacked_test_pred = stacked_prediction(loaded_models, stacked_model, test_images_normalized)
        
        stacked_test_pred_probs = stacked_model.predict_proba(stacked_dataset(loaded_models, test_images_normalized))
        stacked_metrics = evaluate_stack_model(epoch, num_clusters, 'stacked_model', stacked_test_pred, stacked_test_pred_probs, test_labels, result_out, 'stacked_model', categories, time_taken.total_seconds())
        metrics_list.append(stacked_metrics)

    # Save all metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(result_folder, f'combined_metrics_num_clusters_{num_clusters}.csv'), index=False)
    print(f"Combined metrics saved to {os.path.join(result_folder, f'combined_metrics_num_clusters_{num_clusters}.csv')}")

if __name__ == "__main__":
    main()