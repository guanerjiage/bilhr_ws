#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import pandas as pd
import operator
import json

def unicode_convert(input):
    if isinstance(input, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [unicode_convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input    

current_id = 0

class Node(object):

    def __init__(self, parent, dataset, id=None):
        global current_id
        if id is None:
            self.id = current_id # ID of the node
            current_id += 1
        else:
            self.id = id
            current_id = id + 1
        self.dataset = dataset # dataset belongs to this node
        self.parent = parent # parent node
        self.output = None # leaf node output
        self.children = [] # all children
        self.feature = None # feature type to split into branches
        self.attr = None # feature that the node represents
        self.prob = [1.0] # probability that this node is chosen

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False
    
    def is_root(self):
        if self.attr is None:
            return True
        return False

    def find_child(self, attr):
        if self.is_leaf():
            return -1
        prob_list = []
        for child in self.children:
            prob_list.append(child.prob)
            if child.attr == attr:
                return child
        # No child with attr found, return None
        return None

    def __str__(self):
        if self.is_root():
            output = "Root id %d: feature %s, split probabilities %s, children id: " % (self.id, str(self.feature), str(self.prob))
            for child in self.children:
                output += "%d, " % child.id
        elif self.is_leaf():
            output = "Leaf id %d: parent id %d, attr: %s, feature %s, output: %s, output probabilities: %s" % (self.id, self.parent.id, str(self.attr), str(self.feature), str(self.output), str(self.prob))
            output += ", dataset: %s" % str(self.dataset)
        else:
            output = "Branch id %d: parent id %d, attr: %s, feature %s, split probabilities %s, children id: " % (self.id, self.parent.id, str(self.attr), str(self.feature), str(self.prob))
            for child in self.children:
                output += "%d, " % child.id        

        return output

    # convert the node infomation to a dict
    def convert_to_dict(self):
        output = {}
        output["id"] = self.id
        output["dataset"] = self.dataset
        if self.is_root():
            output["parent"] = None
        else:
            output["parent"] = self.parent.id
        output["output"] = self.output
        output["children"] = [child.id for child in self.children]
        output["feature"] = self.feature
        output["attr"] = self.attr
        output["prob"] = self.prob
        return output

    def info_tree_recursive(self):
        # leaf node
        if self.is_leaf():
            return "%s->%s, " % (self.attr, str(self.output))
        # root node
        if self.is_root():
            output = "%s: {" % (self.feature)
            for child in self.children:
                output += child.info_tree_recursive()
            if output[-1] == " ":
                output = output[:-2]
            output += "}"
            return output
        # normal branch node
        # print(self.feature, self.attr)
        output = "%s->{%s: {" % (self.attr, self.feature)
        for child in self.children:
            output += child.info_tree_recursive()
        if output[-1] == " ":
            output = output[:-2]
        output += "}}, "
        return output

class DecisionTree(object):
    
    # the last column of dataset should be the label, others are features
    def __init__(self, dataset=None, ignored_feature=[], threshold=0.0):
        self.root = None
        self.node_list = []

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = []

        self.threshold = threshold
        self.ignored_feature = ignored_feature

        self.dataset_len = len(self.dataset)
        # set tree_updated to True when call build_tree, when dataset modified, set to False 
        # (be aware when directly modify the dataset without function call)
        self.tree_updated = False

    # append one sample to the dataset
    def append_data(self, data):
        self.dataset.append(data)
        self.dataset_len += 1
        self.tree_updated = False

    # append a sequence of samples to the dataset
    def append_dataset(self, dataset):
        self.dataset.extend(dataset)
        self.dataset_len += len(dataset)
        self.tree_updated = False
        
    # calculate entropy for a probability distribution
    def calc_entropy(self, probabilities):
        prob = np.array(probabilities)
        prob = prob[prob > 0] # eliminate 0 terms
        ent = -prob * np.log2(prob) # -sum(p*log2(p))
        return np.sum(ent)

    # calculate marginal entropy for each feature type in the dataset, or the selected features
    def calc_entropy_dataset(self, dataset, selected_features=[]):
        num_data = len(dataset)
        if num_data == 0:
            return 0
        
        entropy_marg_list = []
        num_feature_and_label = len(dataset[0])
        # choose features need to be considered, for example [-1] means only calculate entropy for the label
        if len(selected_features) == 0:
            selected_features = list(range(num_feature_and_label))
        
        for index_feature in selected_features:
            # placeholders
            feature_count = {}
            feature_list = []
            # calculate probability of features (or label) of current type
            for index_data in range(num_data):
                current_feature = dataset[index_data][index_feature]
                if current_feature not in feature_list:
                    feature_list.append(current_feature)
                    feature_count[current_feature] = 1.0
                else:
                    feature_count[current_feature] += 1.0
            feature_count = [v for v in feature_count.values()] # convert dict to list (probability don't need feature name)
            feature_prob = feature_count / np.sum(feature_count) # calculate probability
            entropy_marg_list.append(self.calc_entropy(feature_prob)) # calculate entropy and append to list

        return np.array(entropy_marg_list)
    
    # find the best feature to split the current branch
    def choose_feature_to_split(self, dataset, ignored_feature=[]):
        # placeholder
        best_gain = -1.0
        best_feature = -1
        best_dataset_splited = {}
        best_prob_subset_dict = {}

        num_data = len(dataset)
        if num_data == 0:
            return best_feature, best_gain
        
        num_feature = len(dataset[0]) - 1
        entropy_marg_label = self.calc_entropy_dataset(dataset, [-1])[0] # H(Y)
        
        # branches may not need all features to be traversed
        selected_features = list(range(num_feature))
        for feature_to_remove in ignored_feature:
            selected_features.remove(feature_to_remove)
        # calculate infomation gain 
        for index_feature in selected_features:
            entropy_cond_label_feature = 0.0 # H(Y|Xi)
            entropy_split = 0.0 # H(Xi)
            prob_subset_dict = {} # store probability of each subset
            dataset_splited = self.split_dataset(dataset, index_feature)
            for (key, subset) in dataset_splited.items():
                num_subset = len(subset)                
                prob_subset = num_subset / float(num_data) # p(Xi=x)
                prob_subset_dict[key] = prob_subset
                entropy_cond_subset = self.calc_entropy_dataset(subset, [-1]) # H(Y|Xi=x)
                entropy_cond_label_feature += prob_subset * entropy_cond_subset # p(Xi=x) * H(Y|Xi=x)
                entropy_split += -prob_subset * np.log2(prob_subset) # -p(Xi=x) * log2(p(Xi=x))
                # print("num_subset", num_subset)
                # print("prob_subset", prob_subset)
                # print("entropy_cond_subset", entropy_cond_subset)
                # print("entropy_cond_label_feature", entropy_cond_label_feature)
                # print("entropy_split", entropy_split)

            entropy_gain = entropy_marg_label - entropy_cond_label_feature # g(Y, Xi) = H(Y) - H(Y|Xi)
            # print(dataset_splited)
            # print(entropy_gain)
            # print(index_feature)
            if entropy_split == 0:
                entropy_gain_ratio = 1.0
            else:
                entropy_gain_ratio = entropy_gain / entropy_split # g(Y, Xi) / H(Xi)
            # print(index_feature, entropy_gain, entropy_gain_ratio)
            # find max gain
            if entropy_gain_ratio > best_gain:
                best_gain = entropy_gain_ratio
                best_feature = index_feature
                best_dataset_splited = dataset_splited
                best_prob_subset_dict = prob_subset_dict
        
        return best_feature, best_dataset_splited, best_gain, best_prob_subset_dict

    # split dataset along a feature
    def split_dataset(self, dataset, index_feature):
        subset = {}
        for data in dataset:
            subset.setdefault(data[index_feature], []).append(data)
        return subset
    
    # decide the output of a leaf, return labels and counts in descending order
    def leaf_decision(self, labels):
        label_count = {}
        label_list = []
        # count samples belong to each class
        for item in labels:
            if item not in label_list:
                label_list.append(item)
                label_count[item] = 1
            else:
                label_count[item] += 1
        # sort count and find the class with largest count
        label_count_sorted = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return label_count_sorted
    
    # build the decision tree with dataset
    def build_tree(self):
        if self.dataset_len == 0:
            print("dataset is empty!")
            return
        self.node_list[:] = []        
        self.root = self.build_tree_recursive(dataset=self.dataset, parent=None, attr=None, ignored_feature=self.ignored_feature, threshold=self.threshold)
        self.tree_updated = True

    # build the decision tree recursively
    def build_tree_recursive(self, dataset, parent, attr=None, ignored_feature=[], threshold=0.0):
        num_feature = len(dataset[0]) - 1
        labels = [data[-1] for data in dataset]
        
        # all data in the dataset belong to one class, return the class (leaf node)
        if labels.count(labels[0]) == len(labels):
            leaf = Node(parent, dataset)
            leaf.output = [labels[0]]
            leaf.feature = None
            leaf.attr = attr
            leaf.prob = [1.0]
            if parent is not None:
                parent.children.append(leaf)
                parent.dataset = None
            self.node_list.append(leaf)
            return leaf
        
        # all features are considered, decide the class (leaf node)
        if len(ignored_feature) == num_feature:
            output = self.leaf_decision(labels)
            leaf = Node(parent, dataset)
            leaf.feature = None
            leaf.attr = attr
            leaf.output = []
            output_count = []
            for item in output:
                leaf.output.append(item[0])
                output_count.append(item[1])            
            output_prob = output_count / np.sum(output_count, dtype=np.float)
            leaf.prob = output_prob.tolist()
            if parent is not None:
                parent.children.append(leaf)
                parent.dataset = None
            self.node_list.append(leaf)
            return leaf

        best_feature, best_dataset_splited, best_gain, best_prob_subset = self.choose_feature_to_split(dataset, ignored_feature)
        # print("Split feature: ", best_feature, best_gain, best_prob_subset)

        # gain smaller than a threshold, cut the branches (leaf node)
        if best_gain <= threshold:
            output = self.leaf_decision(labels)
            leaf = Node(parent, dataset)
            leaf.feature = None
            leaf.attr = attr
            leaf.output = []
            output_count = []
            for item in output:
                leaf.output.append(item[0])
                output_count.append(item[1])            
            output_prob = output_count / np.sum(output_count, dtype=np.float)
            leaf.prob = output_prob.tolist()
            if parent is not None:
                parent.children.append(leaf)
                parent.dataset = None
            self.node_list.append(leaf)
            return leaf

        # node with branches
        branch = Node(parent, dataset)
        branch.output = None
        branch.feature = best_feature
        branch.attr = attr
        branch.prob = best_prob_subset.values()
        if parent is not None:
            parent.children.append(branch)
            parent.dataset = None
        self.node_list.append(branch)

        ignored_feature_new = ignored_feature + [best_feature]
        # print("ignored_feature_new", ignored_feature_new)
        feature_values = best_dataset_splited.keys()
        for key in feature_values:
            _ = self.build_tree_recursive(best_dataset_splited[key], branch, key, ignored_feature_new, threshold)
        return branch

    # prediction based on the current tree
    def predict(self, feature, root=None):
        if root is None:
            root = self.root
        if not self.tree_updated:
            print("Decision Tree not actual, result might be incorrect!")
        current_node = root
        
        while current_node.output is None:
            current_attr = feature[current_node.feature]
            next_node = current_node.find_child(current_attr)
            if next_node is None:
                return None, None
            # TODO strange bug, solved hopefully
            if next_node == -1:
                print(current_node)
                print(current_node.parent)
            current_node = next_node
        return current_node.output, current_node.prob

    def __str__(self):
        if self.root is not None:
            return self.root.info_tree_recursive()
        return "Tree empty"

    def save_tree(self, filename):
        output = []
        for node in self.node_list:
            node_dict = node.convert_to_dict()
            output.append(node_dict)
        with open(filename, "w") as f:
            json_str = json.dumps(output, indent=4)
            f.write(json_str)
            f.write("\n")
            print("Decision tree saved")

    def load_tree(self, filename):
        with open(filename, "r") as f:
            json_str = f.read()
            node_dict_list = unicode_convert(json.loads(json_str))
        # print(node_dict_list)        
        self.node_list = []
        self.dataset = []
        for node_dict in node_dict_list:
            new_node = Node(node_dict["parent"], node_dict["dataset"], node_dict["id"])
            new_node.output = node_dict["output"]
            new_node.prob = node_dict["prob"]
            new_node.feature = node_dict["feature"]
            new_node.attr = node_dict["attr"]
            new_node.children = node_dict["children"]
            self.node_list.append(new_node)
        for node in self.node_list:
            # append dataset in lives to tree
            if node.dataset is not None:
                self.append_dataset(node.dataset)
            # find parent and children node
            for node_found in self.node_list:
                if node_found.id == node.parent:
                    node.parent = node_found
                if node.children is not None:
                    for child_index in range(len(node.children)):
                        if node_found.id == node.children[child_index]:
                            node.children[child_index] = node_found
            # find root
            if node.is_root():
                self.root = node
        print("Decision tree load finish")    

if __name__ == "__main__":
    dataset = pd.read_csv("dataset.csv")
    dataset.info()
    dataset_list = dataset.values.tolist()
    # print(dataset_list)

    dt = DecisionTree(dataset_list, [], 0.0)
    print("Dataset length: ", dt.dataset_len)
    dt.build_tree()
    print(dt.root)
    print(dt)
    print(dt.predict(['Youth', 'Yes', 'No', 'Normal']))

    print("====================================\n")

    dataset3 = pd.read_csv("dataset3.csv")
    dataset3.info()
    dataset3_list = dataset3.values.tolist()
    # print(dataset3_list)

    dt3 = DecisionTree(dataset3_list, [], 0.0)
    print("Dataset3 length: ", dt3.dataset_len)
    print(dt3.dataset)
    dt3.build_tree()
    print(dt3.root)
    print(dt3)
    print(dt3.predict(['Youth', 'No', 'No', 'Good']))
    print(dt3.predict(['Youth', 'Yes', 'YES', 'Good']))

    print("------------------------------------\n")

    dt3.save_tree("test.json")
    dt3.load_tree("test.json")

    print(dt3.dataset)
    print(dt3.root)
    print(dt3)    
    print(dt3.predict(['Youth', 'No', 'No', 'Good']))
    print(dt3.predict(['Youth', 'Yes', 'YES', 'Good']))

    print("---------------------------------\n")
    dt3.build_tree()
    print(dt3.root)
    print(dt3)
    print(dt3.predict(['Youth', 'No', 'No', 'Good']))
    print(dt3.predict(['Youth', 'Yes', 'YES', 'Good']))

    # test an extreme case: split has no gain
    print("==================================\n")
    dataset4 = [[5, 0, 'down', -1], [5, 0, 'down', 0], [5, 0, 'down', -1], [5, 1, 'down', 0], [5, 0, 'down', 0], [5, 1, 'down', -1]]
    dt4 = DecisionTree(dataset4)
    dt4.build_tree()
    print(dt4)
    print(dt4.predict([5, 0, 'down']))
    


