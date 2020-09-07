import pandas as pd
import numpy as np
import os
def create_submission(preds, submission_name):
    submission_path = "./submissions/"
    sub = pd.read_csv("./data/sample_submission.csv")
    sub["Business_Sourced"] = preds
    sub.to_csv(os.path.join(submission_path, submission_name + ".csv"),index = False)
    print(f"saved_submission: ", submission_name)


if __name__ == "__main__":
    test_size = pd.read_csv("./data/sample_submission.csv").shape[0]
    preds = np.zeros(test_size)
    create_submission(preds, "all_zeros")
    preds = np.ones(test_size)
    create_submission(preds, "all_ones")