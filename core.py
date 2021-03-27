import argparse
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from utils.watch import watch


def execute(command):
    p = subprocess.Popen(command, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            os.kill(p.pid, signal.SIGINT)
        except OSError:
            pass
        p.wait()


def prepare_parser():
    parser = argparse.ArgumentParser(
        description='The core script of experiment management.')
    subparsers = parser.add_subparsers(dest='command')

    add_env_parser(subparsers)
    add_watch_parser(subparsers)

    return parser


def add_env_parser(subparsers):
    parser = subparsers.add_parser(
        'env', description='Environment Management.')
    parser.add_argument(
        'action', nargs='?', choices=['prepare', 'enter', 'stop'],
        default='enter')
    parser.add_argument('-b', '--build', action='store_true', default=False)
    parser.add_argument('--root', action='store_true', default=False)


def env(args):
    execute('echo "UID=$(id -u)\nGID=$(id -g)\nUSER_NAME=$(whoami)" > ./.env')
    if args.action == 'prepare':
        command = 'docker-compose up -d'
        if args.build:
            command += ' --build'
    elif args.action == 'enter':
        if args.root:
            command = 'docker-compose exec -u root playground zsh'
        else:
            command = 'docker-compose exec playground zsh'
    elif args.action == 'stop':
        command = 'docker-compose stop'
    else:
        raise NotImplementedError
    execute(command)


def add_watch_parser(subparsers):
    """
    This script will run <command> when and only when the condition is true.
    Usage:
        # python core.py watch <type>:<target> "<command>"  [--gap <seconds>] [--reverse]
    For example:
    1. Waiting until the target file exists:
        # python watch.py f:test "ls -l" --gap 1
    2. Waiting until the process pid:1 quits:
        # python watch.py p:1 "ls -l" --gap 1
    """
    parser = subparsers.add_parser('watch', description='Wather.')
    parser.add_argument("target")
    parser.add_argument("command")
    parser.add_argument("--gap", default=300, type=int)
    parser.add_argument("--reverse", action="store_true")


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'env':
        env(args)
    elif args.command == 'watch':
        watch(args)
    else:
        pass