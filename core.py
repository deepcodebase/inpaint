import argparse
import os
import signal
import subprocess
from pathlib import Path

from utils.watch import watch, add_watch_args
from utils.env import Env


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
        description="The core script of experiment management."
    )
    subparsers = parser.add_subparsers(dest="command")

    add_env_parser(subparsers)
    add_watch_parser(subparsers)

    return parser


def add_env_parser(subparsers):
    parser = subparsers.add_parser(
        "env", description="Environment Management.")
    parser.add_argument("action", nargs="?", default="enter")
    parser.add_argument("-b", "--build", action="store_true", default=False)
    parser.add_argument("--root", action="store_true", default=False)
    parser.add_argument("--code_root", type=str, default='.')


def env(args):
    e = _set_env()
    _create_log_symlink(Path(e['LOG_ROOT']['val']))

    if args.action == "prepare":
        command = "docker-compose up -d"
        if args.build:
            command += " --build"
    elif args.action == "enter":
        if args.root:
            command = "docker-compose exec -u root lab zsh"
        else:
            command = "docker-compose exec lab zsh"
    elif args.action == "stop":
        command = "docker-compose stop"
    else:
        command = f"docker-compose {args.action}"
    execute(command)


def _set_env():
    e = Env('.env')
    if 'UID' not in e or 'GID' not in e or 'USER_NAME' not in e:
        e['UID'] = os.getuid()
        e['GID'] = os.getgid()
        e['USER_NAME'] = os.getlogin()
    e['CODE_ROOT'] = args.code_root
    if 'PROJECT' not in e:
        project = input('Give a project name: ').strip()
        e['PROJECT'] = str(project)
    if 'LOG_ROOT' not in e:
        log_root = Path(input(
            'Input the log dir (will be mounted to /outputs): '))
        log_root.expanduser().mkdir(exist_ok=True, parents=True)
        e['LOG_ROOT'] = str(log_root)
    if 'DATA_ROOT' not in e:
        data_root = Path(input(
            'Input the data dir (will be mounted to /data): '))
        data_root.expanduser().mkdir(exist_ok=True, parents=True)
        e['DATA_ROOT'] = str(data_root)
    e.save()
    return e


def _create_log_symlink(log_path):
    link_path = Path('outputs')
    if not link_path.exists() and not link_path.is_symlink():
        link_path.symlink_to(
            log_path.expanduser(), target_is_directory=True)


def add_watch_parser(subparsers):
    parser = subparsers.add_parser("watch", description="Wather.")
    add_watch_args(parser)


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    if args.command == "env":
        env(args)
    elif args.command == "watch":
        watch(args)
    else:
        pass
