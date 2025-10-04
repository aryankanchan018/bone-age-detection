import os
import pandas as pd


def load_and_prepare_data(dataframe_direction):
    """
    Load and prepare training and testing dataframes.
    
    Args:
        dataframe_direction: Path to the dataset directory
    
    Returns:
        Tuple of (training_dataframe, testing_dataframe)
    """
    # Load training data
    training_dataframe = pd.read_csv(
        os.path.join(dataframe_direction, "boneage-training-dataset.csv")
    )
    training_dataframe["path"] = training_dataframe["id"].map(
        lambda x: os.path.join(
            dataframe_direction,
            "boneage-training-dataset",
            "boneage-training-dataset",
            f"{x}.png"
        )
    )
    training_dataframe["imagepath"] = training_dataframe["id"].map(lambda x: f"{x}.png")
    training_dataframe["gender"] = training_dataframe["male"].map(
        lambda x: "male" if x else "female"
    )
    training_dataframe["gender_encoded"] = training_dataframe["gender"].map(
        lambda x: 1 if x == "male" else 0
    )
    training_dataframe["boneage_category"] = pd.cut(training_dataframe["boneage"], 10)
    
    # Normalize bone age
    boneage_std = 2 * training_dataframe["boneage"].std()
    boneage_mean = training_dataframe["boneage"].mean()
    training_dataframe["norm_age"] = (
        training_dataframe["boneage"] - boneage_mean
    ) / boneage_std

    # Load testing data
    testing_dataframe = pd.read_csv(
        os.path.join(dataframe_direction, "boneage-test-dataset.csv")
    )
    testing_dataframe["path"] = testing_dataframe["Case ID"].map(
        lambda x: os.path.join(
            dataframe_direction,
            "boneage-test-dataset",
            "boneage-test-dataset",
            f"{x}.png"
        )
    )
    testing_dataframe["imagepath"] = testing_dataframe["Case ID"].map(lambda x: f"{x}.png")
    testing_dataframe["gender"] = testing_dataframe["Sex"].map(
        lambda x: "male" if x == "M" else "female"
    )

    print(f"Training samples: {len(training_dataframe)}")
    print(f"Testing samples: {len(testing_dataframe)}")
    
    return training_dataframe, testing_dataframe
