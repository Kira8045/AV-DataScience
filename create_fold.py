from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def split_balanced(data, target, test_size=0.2):
    
    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data.loc[ix_train,:]
    X_test = data.loc[ix_test,:]

    X_test.to_csv("./data/val.csv", index = False)
    X_train.to_csv("./data/train.csv", index = False)
    print("saved balanced split of train data")

def create_folds_KFold(data, save_name = "./data/train_fold.csv", folds = 5, target_feature = "Business_Sourced",):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    X = data.drop(target_feature, axis = 1)
    y = data[target_feature]
    data.loc[:, "Fold"] = -1
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        data.loc[test_index, "Fold"] = i
    data.to_csv(save_name, index = False)
    print("save kFold train Data in: ", save_name.split("/")[-1])
if __name__ == "__main__":

    data = pd.read_csv("./data/Train_orig.csv")
    # create_folds_KFold(data)
    split_balanced(data, data[:]["Business_Sourced"])

    