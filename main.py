import argparse
import os
from variables.default import DefaultTraining, DefaultPruning, DefaultEvaluation, DefaultFactorization

def main():    
    parser = argparse.ArgumentParser(description="Deep Learning ResNet Image Classification CIFAR10")
    parser.add_argument("mode", choices=["train", "evaluate","binary_train","pruning_train","factorization_train"], help="Mode to run the script")
    parser.add_argument("--data_path", type=str, default=DefaultTraining().data_path, help="Path to the dataset")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=DefaultTraining().epochs, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=DefaultTraining().weight_decay, help="Weight decay for regularization")
    parser.add_argument("--batch_size", type=int, default=DefaultTraining().batch_size, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=DefaultTraining().learning_rate, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=DefaultTraining().save_path, help="Path to save the model")

    # Pruning 
    parser.add_argument("--amount", type=float, default=DefaultPruning().amount, help="amount for global unstructured pruning")

    # Evaluation arguments
    parser.add_argument("--model_path", type=str, default=DefaultEvaluation().model_path, help="Path to the trained model")

    # Factorization
    parser.add_argument("--use_depthwise", action='store_true', help="Use depthwise separable convolutions")
    parser.add_argument("--use_grouped", action='store_true', help="Use grouped factorization on convolutions")
    parser.add_argument("--groups", type=int, default=DefaultFactorization().groups, help="Group size for grouped factorization")

    args = parser.parse_args()

    if args.mode == "train":
        os.system(f"python src/train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --save_path {args.save_path}")
    elif args.mode == "evaluate":
        os.system(f"python src/evaluate.py --model_path {args.model_path} --batch_size {args.batch_size} --data_path {args.data_path}")

    elif args.mode == "binary_train":
        os.system(f"python src/binary_train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --save_path {args.save_path}")

    elif args.mode == "pruning_train":
        os.system(f"python src/pruning_train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --amount {args.amount}")

    elif args.mode == "factorization_train":
        os.system(f"python src/factorization_train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --use_depthwise --use_grouped --groups {args.groups}")

if __name__ == "__main__":
    main()