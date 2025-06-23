class DefaultTraining:
    data_path = "/opt/img/effdl-cifar10/"
    epochs = 10
    weight_decay = 0.0005
    batch_size = 64
    learning_rate = 0.001
    save_path = "models/you_did_not_set_model_path.pth"

class DefaultPruning:
    amount = 0.2

class DefaultEvaluation:
    model_path = "models/test.pth"

class DefaultFactorization:
    groups = 2
