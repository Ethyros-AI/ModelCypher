import argparse
import sys
from modelcypher.interfaces.cli import train_cli, inspect_cli, dynamics_cli

def main():
    parser = argparse.ArgumentParser(description="ModelCypher CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Register subcommands
    train_cli.register(subparsers)
    inspect_cli.register(subparsers)
    dynamics_cli.register(subparsers)
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_cli.run(args)
    elif args.command == "inspect":
        inspect_cli.run(args)
    elif args.command == "dynamics":
        dynamics_cli.run(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
