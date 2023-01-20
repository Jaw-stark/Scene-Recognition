import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
#from scipy import stats
from pathlib import Path, PureWindowsPath
import warnings
warnings.filterwarnings("ignore")

def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img,stride,size):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = [cv2.KeyPoint(x+int(size/2), y+int(size/2),size) for y in range(0, img.shape[0], stride)
                                        for x in range(0, img.shape[1], stride)]
    kp, dense_feature = sift.compute(img,kp1)
    return dense_feature


def get_tiny_image(img, output_size):
    w, h = output_size
    N = len(img)
    feature = np.zeros(N)
    for imgr in range(N):
        image = cv2.imread(img[imgr])
        feat = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        fmean = np.mean(feat)
        feature[imgr] = fmean / np.linalg.norm(feat) 
    return feature

#def knn(xTrain, xTest, k=2):
#    nn = NearestNeighbors(n_neighbors=k)
#    distances = nn.fit(xTrain)

def predict_knn(feature_train, label_train, feature_test, k):
    #nn = NearestNeighbors(n_neighbors=k)
    #fitted = nn.fit(feature_train)
    knn = KNeighborsClassifier(k)
    knn.fit(feature_train, label_train)
    label_test_pred = knn.predict(feature_test)
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    k = 3
    feature_train= get_tiny_image(img_train_list, (16,16)).reshape(1500,1)
    feature_test= get_tiny_image(img_test_list, (16,16)).reshape(1500,1)
    
    label_test_pred = predict_knn(feature_train,label_train_list,feature_test,k)
    accuracy = accuracy_score(label_test_list, label_test_pred)
    confusion = confusion_matrix(label_test_list, label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dict_size):
    dense_feature = np.zeros((1,128))
    
    for df in dense_feature_list:

        dense_feature = np.concatenate((dense_feature, df), axis=0)
    set_dense = np.delete(dense_feature, 0, 0)
    
    kmeans = KMeans(n_clusters = 50,n_init=10,max_iter=300).fit(set_dense)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    n_neigh = 1
    # Using Nearest Neighbors
    NN = NearestNeighbors(n_neighbors=n_neigh)
    NN.fit(vocab)
    fits = NN.kneighbors(feature, n_neigh, return_distance=False)
    bow_feature = np.zeros((1, vocab.shape[0]))
    for i in range(0, fits.shape[0]):
        bf = fits[i][0]
        bow_feature[0][bf]+=1
    bow_feature = bow_feature/fits.shape[0]
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    '''Compute for training data X_train'''
    stride = 20
    size = 20
    # call the compute dsift function
    dense_flist_train = []
    for img in img_train_list:
        im = cv2.imread(img,0)
        
        dense_feat = compute_dsift(im, stride, size)
        dense_flist_train.append(dense_feat)
    # Populate and save the vocab
    vocab = build_visual_dictionary(dense_flist_train, 50)
    np.savetxt('knn_bow', vocab, delimiter=',')
    
    vocab_flist_train = []
    for img in img_train_list:
        im = cv2.imread(img,0)
        
        dense_feat = compute_dsift(im, stride, size)
    
        bow_train = compute_bow(dense_feat, vocab)
        vocab_flist_train.append(bow_train)
    vocab_flist_train = np.array(vocab_flist_train).reshape(1500,50)
    ''' Compute for test data X_test'''
    vocab_flist_test = []
    for img in img_test_list:
        im = cv2.imread(img,0)
        
        dense_feat = compute_dsift(im, stride, size)
        bow_test = compute_bow(dense_feat, vocab)
        vocab_flist_test.append(bow_test)
    vocab_flist_test = np.array(vocab_flist_test).reshape(1500,50)
    ''' Compute the training labels y_train and y_test'''
    train_labels_indices = []
    for idx in label_train_list:
        train_labels_indices.append(label_classes.index(idx))
     
    # convert the indicies into arrays
    labels_train = np.array(train_labels_indices).reshape(1500,1)
     
    # For test data
    test_labels_indices = []
    for idx in label_test_list:
        test_labels_indices.append(label_classes.index(idx))
     
    # convert the indicies into arrays
    labels_test = np.array(test_labels_indices).reshape(1500,1)
    
    # Define K
    k = 10
    
    
    # Call the KNN Classifier
    label_test_pred = predict_knn(vocab_flist_train,labels_train, vocab_flist_test,k)
    confusion = confusion_matrix(labels_test, label_test_pred)
    accuracy = accuracy_score(labels_test, label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    svm = LinearSVC()
    svm.fit(feature_train, label_train, n_classes)
    label_test_pred = svm.predict(feature_test)
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    '''Compute for training data X_train'''
    stride = 20
    size = 20
    
    
    # call the compute dsift function
    dense_flist_train = []
    for img in img_train_list:
        im = cv2.imread(img,0)
        
        dense_feat = compute_dsift(im, stride, size)
        dense_flist_train.append(dense_feat)
    # Populate and save the vocab
    vocab = build_visual_dictionary(dense_flist_train, 50)
    np.savetxt('svm_bow', vocab, delimiter=',')
    
    vocab_flist_train = []
    for img in img_train_list:
        im = cv2.imread(img,0)
        
        dense_feat = compute_dsift(im, stride, size)
    
        bow_train = compute_bow(dense_feat, vocab)
        vocab_flist_train.append(bow_train)
    vocab_flist_train = np.array(vocab_flist_train).reshape(1500,50)
    ''' Compute for test data X_test'''
    vocab_flist_test = []
    for img in img_train_list:
        im = cv2.imread(img,0)
        
        dense_feat = compute_dsift(im, stride, size)
   
        bow_test = compute_bow(dense_feat, vocab)
        vocab_flist_test.append(bow_test)
    vocab_flist_test = np.array(vocab_flist_test).reshape(1500,50)
    ''' Compute the training labels y_train and y_test'''
    train_labels_indices = []
    for idx in label_train_list:
        train_labels_indices.append(label_classes.index(idx))
     
    # convert the indicies into arrays
    labels_train = np.array(train_labels_indices).reshape(1500,1)
     
    # For test data
    test_labels_indices = []
    for idx in label_test_list:
        test_labels_indices.append(label_classes.index(idx))
     
    # convert the indicies into arrays
    labels_test = np.array(test_labels_indices).reshape(1500,1)
    
    # Define K
    n_classes = 10
    
    # Call the KNN Classifier
    label_test_pred = predict_svm(vocab_flist_train,labels_train, vocab_flist_test,n_classes)
    confusion = confusion_matrix(labels_test, label_test_pred)
    accuracy = accuracy_score(labels_test, label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    dict_size = 50
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




