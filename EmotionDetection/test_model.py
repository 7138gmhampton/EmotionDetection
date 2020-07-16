import argparse

parser = argparse.ArgumentParser(description='Apply test examples to trained model.')
parser.add_argument('--model', help='The datetime for the model', required=True)
args = parser.parse_args()

print(args.model)
