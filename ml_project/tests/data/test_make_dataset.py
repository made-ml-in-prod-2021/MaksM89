from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.enities.data_params import InputDataset, SplittingParams


def test_load_dataset(dataset_info: InputDataset):
    data = read_data(dataset_info)
    assert len(data.columns) == 14, "Columns count must be 14"
    assert len(data) > 10, "Not enough data"
    assert dataset_info.target_col in data.columns, "No target col in data"


def test_split_dataset(tmpdir, dataset_info: InputDataset):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=42, val_size=val_size,)
    data = read_data(dataset_info)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10
