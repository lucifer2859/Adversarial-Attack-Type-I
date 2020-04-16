import argparse
from mnist import train, transition, train_MLP, attack


def parse_args():
    desc = "Codes based on Tensorflow for the paper: Adversarial Attack Type I: Generating False Positives."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='./results',
                        help='File path of output images')

    parser.add_argument('--dim_z', type=int, default=32, help='Dimension of latent vector')

    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--epochs', type=int, default=600, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--phase', type=str, default='train', help='train/transition/attack/trainMLP')

    parser.add_argument('--classes', type=int, default=10, help='number of classes of the dataset')

    parser.add_argument('--test_index', type=int, default=1, help='image index in test set')

    parser.add_argument('--target_label', type=int, default=8, help='targeted image label for Type I attack')

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    if args.phase == 'train':
        train(args)
    elif args.phase == 'transition':
        transition(args)
    elif args.phase == 'trainMLP':
        train_MLP(args)
    elif args.phase == 'attack':
        attack(args)
