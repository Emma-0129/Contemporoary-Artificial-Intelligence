task_name = "LSTM_Alexnet"
# task_name = "TextLSTM"

config = {
    "dataset": "text",
    "model_name": task_name,
    "vocab_path": "./pretrained/glove/vocab.pkl",
    "gpu": "0",
    "dropout": 0.5,
    "patience": 6,
    "epochs": 30,
    "batch_size": 32,
    "pad_size": 32,
    "learning_rate": 8e-3,
    "save_result": False,
}
