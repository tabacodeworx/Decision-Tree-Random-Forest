import unittest
import DecisionTree as dt
import numpy as np
import time

class DecisionTreePart1Tests(unittest.TestCase):
    """Test tree example, confusion matrix, precision, recall, and accuracy.

    Attributes:
        hand_tree (DecisionTreeNode): root node of the built example tree.
        ht_examples (list(list(int)): features for example tree.
        ht_classes (list(int)): classes for example tree."""

    def setUp(self):
        """Setup test data.
        """

        self.hand_tree = dt.build_decision_tree()
        self.ht_examples = [[1, 0, 0, 0],
                            [1, 0, 1, 1],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [1, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]]
        self.ht_classes = [1, 1, 1, 0, 1, 0, 1, 0]

    def test_hand_tree_accuracy(self):
        """Test accuracy of the tree example.

        Asserts:
            decide return matches true class for all classes.
        """

        for index in range(0, len(self.ht_examples)):
            decision = self.hand_tree.decide(self.ht_examples[index])

            assert decision == self.ht_classes[index]

    def test_confusion_matrix(self):
        """Test confusion matrix for the example tree.

        Asserts:
            confusion matrix is correct.
        """

        answer = [1, 0, 0, 1, 0, 0, 0]
        true_label = [1, 1, 1, 0, 0, 0, 0]
        test_matrix = [[1, 2], [1, 3]]

        assert np.array_equal(test_matrix, dt.confusion_matrix(answer,
                                                               true_label))

    def test_precision_calculation(self):
        """Test precision calculation.

        Asserts:
            Precision matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 0, 0, 0, 0]

        for index in range(0, len(answer)):
            answer[index] = 1
            precision = 1 / (1 + index)

            assert dt.precision(answer, true_label) == precision

    def test_recall_calculation(self):
        """Test recall calculation.

        Asserts:
            Recall matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            recall = (index + 1) / ((index + 1) + (total_count - (index + 1)))

            assert dt.recall(answer, true_label) == recall

    def test_accuracy_calculation(self):
        """Test accuracy calculation.

        Asserts:
            Accuracy matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            accuracy = dt.accuracy(answer, true_label)

            assert accuracy == ((index + 1) / total_count)


class DecisionTreePart2Tests(unittest.TestCase):
    """Tests for Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """

        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv('data.csv')
        self.train_features, self.train_classes = self.dataset

    def test_gini_impurity_max(self):
        """Test maximum gini impurity.

        Asserts:
            gini impurity is 0.5.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 0, 0, 0])

        assert  .500 == round(gini_impurity, 3)

    def test_gini_impurity_min(self):
        """Test minimum gini impurity.

        Asserts:
            entropy is 0.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 1, 1, 1])

        assert 0 == round(gini_impurity, 3)

    def test_gini_impurity(self):
        """Test gini impurity.

        Asserts:
            gini impurity is matched as expected.
        """

        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0])

        assert round(4. / 9., 3) == round(gini_impurity, 3)

    def test_gini_gain_max(self):
        """Test maximum gini gain.

        Asserts:
            gini gain is 0.5.
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 1], [0, 0, 0]])

        assert .500 == round(gini_gain, 3)

    def test_gini_gain(self):
        """Test gini gain.

        Asserts:
            gini gain is within acceptable bounds
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 0], [1, 0, 0]])

        assert 0.056 == round(gini_gain, 3)

    def test_gini_gain_restaurant_patrons(self):
        """Test gini gain using restaurant patrons.

        Asserts:
            gini gain rounded to 3 decimal places matches as expected.
        """

        gain_patrons = dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_patrons'])

        assert round(gain_patrons, 3) == 0.278

    def test_gini_gain_restaurant_type(self):
        """Test gini gain using restaurant food type.

        Asserts:
            gini gain is 0.
        """

        gain_type = round(dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_food_type']), 2)

        assert gain_type == 0.00



if __name__ == '__main__':
    unittest.main()
