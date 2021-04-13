import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
import time

class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label
		

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """
    decision_tree_root = DecisionNode(None, None, lambda a: a[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    a4 = decision_tree_root.right = DecisionNode(None, None, lambda a: a[3] == 1)
    a2 = a4.left = DecisionNode(None, None, lambda a: a[1] == 0)
    a2.left = DecisionNode(None, None, None, 1)
    a2.right = DecisionNode(None, None, None, 0)
    a3 = a4.right = DecisionNode(None, None, lambda a: a[2] == 0)
    a3.left = DecisionNode(None, None, None, 1)
    a3.right = DecisionNode(None, None, None, 0)

    return decision_tree_root

def ttSplit(features, classes, prctTrain):
    n = features.shape[0]
    train_ind = random.sample(list(np.arange(n)), round(prctTrain*n))
    features_train = features[train_ind, :]
    classes_train = classes[train_ind]
    features_test = np.delete(features, train_ind, 0)
    classes_test = np.delete(classes, train_ind, 0)
    return features_train, classes_train, features_test, classes_test


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    confMat = np.zeros((2, 2))

    confMat[0, 0] = sum([classifier_output[i] == 1 and true_labels[i] == 1 for i in range(len(true_labels))])
    confMat[0, 1] = sum([classifier_output[i] == 0 and true_labels[i] == 1 for i in range(len(true_labels))])
    confMat[1, 0] = sum([classifier_output[i] == 1 and true_labels[i] == 0 for i in range(len(true_labels))])
    confMat[1, 1] = sum([classifier_output[i] == 0 and true_labels[i] == 0 for i in range(len(true_labels))])

    return confMat


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """
    true_positive = sum([classifier_output[i] == 1 and true_labels[i] == 1 for i in range(len(true_labels))])
    false_positive = sum([classifier_output[i] == 1 and true_labels[i] == 0 for i in range(len(true_labels))])
    precision = true_positive/ (true_positive + false_positive)
    return precision


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """
    true_positive = sum([classifier_output[i] == 1 and true_labels[i] == 1 for i in range(len(true_labels))])
    false_negative = sum([classifier_output[i] == 0 and true_labels[i] == 1 for i in range(len(true_labels))])
    recall = true_positive / (true_positive + false_negative)
    return recall


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    true_positive = sum([classifier_output[i] == 1 and true_labels[i] == 1 for i in range(len(true_labels))])
    true_negative = sum([classifier_output[i] == 0 and true_labels[i] == 0 for i in range(len(true_labels))])
    accuracy = (true_positive + true_negative)/len(true_labels)
    return accuracy


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    pk = sum(class_vector)/len(class_vector)
    IG = 1. - (pk**2 + (1. - pk)**2)
    return IG


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    # def B(p):
    #     IE = 0.
    #     if p > 0. and p < 1.: IE = - (p*np.log2(p) + (1.-p)*np.log2(1.-p))
    #     return IE
    # pk = sum(previous_classes) / len(previous_classes)
    # H_T = B(pk)
    # H_Ta = 0
    # for current_class in current_classes:
    #     pa = len(current_class) / len(previous_classes)
    #     pka = sum(current_class) / len(current_class)
    #     H_Ta += pa * B(pka)
    H_T = gini_impurity(previous_classes)
    H_Ta = 0
    for current_class in current_classes:
        pa = len(current_class) / len(previous_classes)
        if pa > 0: H_Ta += pa * gini_impurity(current_class)
    return H_T - H_Ta

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        def calFeatureGini(feature, klass, splitPt):
            previous_classes = klass.tolist()
            current_classes = [klass[feature <= splitPt].tolist(), klass[feature > splitPt].tolist()]
            return gini_gain(previous_classes, current_classes)

        def calFeatureMaxGini(feature, klass, bins):
            splitPts = np.linspace(min(feature), max(feature), bins)
            bestGini = -float('inf')
            bestSplit = 0
            for splitPt in splitPts:
                gini = calFeatureGini(feature, klass, splitPt)
                if gini > bestGini:
                    bestGini = gini
                    bestSplit = splitPt
            return bestGini, bestSplit

        def calMaxFeatureGini(featuresX, klass, bins):
            bestGini = -float('inf')
            bestSplit = 0
            bestFeature = 0
            for ifeature in range(featuresX.shape[1]):
                feature = featuresX[:, ifeature]
                featureGini, featureSplit = calFeatureMaxGini(feature, klass, bins)
                if featureGini > bestGini:
                    bestGini = featureGini
                    bestSplit = featureSplit
                    bestFeature = ifeature
            return bestFeature, bestSplit

        def plurality(klass):
            ans = int(0)
            if sum(klass) >= 0.5*len(klass): ans = int(1)
            return ans

        def getDecisionRoot(featuresX, klass, parentKlass, curr_depth, bins):
            if featuresX.size == 0:
                return plurality(parentKlass)
            elif all(klass == 1) or all(klass == 0) or curr_depth == depth:
                return plurality(klass)
            else:
                curr_depth += 1
                featureInd, splitVal = calMaxFeatureGini(featuresX, klass, bins)
                tree = DecisionNode(None, None, lambda a: a[featureInd] <= splitVal)

                yesLabel = featuresX[:, featureInd] <= splitVal
                features1 = featuresX[yesLabel]
                klass1 = klass[yesLabel]
                subtree1 = getDecisionRoot(features1, klass1, klass, curr_depth, bins)
                if isinstance(subtree1, int):
                    tree.left = DecisionNode(None, None, None, subtree1)
                else:
                    tree.left = subtree1

                noLabel = np.logical_not(yesLabel)
                features0 = featuresX[noLabel]
                klass0 = klass[noLabel]
                subtree0 = getDecisionRoot(features0, klass0, klass, curr_depth, bins)
                if isinstance(subtree0, int):
                    tree.right = DecisionNode(None, None, None, subtree0)
                else:
                    tree.right = subtree0

                return tree

        curr_depth = 0
        depth = self.depth_limit
        if depth == float('inf'): depth = 5
        bins = 10
        return getDecisionRoot(features, classes, classes, curr_depth, bins)


    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for index in range(0, len(features)):
            decision = self.root.decide(features[index])
            class_labels += [decision]
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    features = dataset[0]
    classes = dataset[1]
    testSize = len(classes) // k
    subsets = []
    for j in range(k-1):
        sampleInd = random.sample(range(len(classes)), testSize)
        subsets.append((features[sampleInd], classes[sampleInd]))
        remSampleInd = list( set(range(len(classes))).difference(set(sampleInd)) )
        features = features[remSampleInd]
        classes = classes[remSampleInd]
    subsets.append((features, classes))
    folds = []
    for j in range(k):
        training_set_features = []
        training_set_classes = []
        test_set = []
        for i in range(k):
            if i == j:
                test_set = subsets[i]
            else:
                try:
                    training_set_features = np.concatenate((training_set_features, subsets[i][0]), axis=0)
                    training_set_classes = np.concatenate((training_set_classes, subsets[i][1]), axis=0)
                except:
                    training_set_features = subsets[i][0]
                    training_set_classes = subsets[i][1]
        folds.append(((training_set_features, training_set_classes), test_set))
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        self.forest = []
        self.attrInd = []
        for i in range(self.num_trees):
            forest = DecisionTree(self.depth_limit)
            exampleSampleInd = random.choices(range(len(classes)), k=int(self.example_subsample_rate*len(classes)))
            attrSampleInd = random.sample(range(len(features[0])), k=int(self.attr_subsample_rate*len(features[0])))
            featureSample = features[exampleSampleInd]
            featureSample = featureSample[:, attrSampleInd]
            classesSample = classes[exampleSampleInd]
            forest.fit(featureSample, classesSample)
            self.forest.append(forest)
            self.attrInd.append(attrSampleInd)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """
        tree_labels = []
        for i in range(self.num_trees):
            tree = self.forest[i]
            featureSamp = features[:, self.attrInd[i]]
            tree_labels.append(tree.classify(featureSamp))
        class_labels = [np.around(sum([tree_labels[i][j] for i in range(self.num_trees)]) /self.num_trees) for j in range(len(features))]
        return class_labels

def modelSummary(model, features_train, classes_train, features_test, classes_test):
    predClassTrain = model.classify(features_train)
    precisionTrain = precision(predClassTrain, classes_train)
    recallTrain = recall(predClassTrain, classes_train)
    print('Train accuracy = ', accuracy(predClassTrain, classes_train), end="\n")
    print('Train precision = ', precisionTrain, end="\n")
    print('Train recall = ', recallTrain, end="\n")
    print('Train F1-score = ', 2. / (1 / recallTrain + 1 / precisionTrain), end="\n\n")

    predClassTest = model.classify(features_test)
    precisionTest = precision(predClassTest, classes_test)
    recallTest = recall(predClassTest, classes_test)
    print('Test accuracy = ', accuracy(predClassTest, classes_test), end="\n")
    print('Test precision = ', precisionTest, end="\n")
    print('Test recall = ', recallTest, end="\n")
    print('Test F1-score = ', 2. / (1 / recallTest + 1 / precisionTest), end="\n")




num_trees = 5
depth_limit = 3
example_subsample_rate = 0.5
attr_subsample_rate = 0.5
forest = RandomForest(num_trees, depth_limit, example_subsample_rate, attr_subsample_rate)
features, classes = load_csv('data.csv')
features_train, classes_train, features_test, classes_test = ttSplit(features, classes, 0.8)
forest.fit(features_train, classes_train)
modelSummary(forest, features_train, classes_train, features_test, classes_test)

