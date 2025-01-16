from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
import os
import argparse
from dotenv import load_dotenv


def upload_repo(*, host: str, channel: str, username: str, password: str):
    repo_dir = Path(__file__).parent / "repo"
    arches = [d for d in repo_dir.iterdir() if d.is_dir()]
    for arch in arches:
        repodata_path = arch / "repodata.json"
        if not repodata_path.exists():
            print(f"Skipping {arch} as it does not contain repodata.json")
            continue

        url = f"{host}/channels/{channel}/{arch.name}/repodata.json"
        with open(repodata_path, "rb") as f:
            response = requests.post(
                url, files={"file": f}, auth=HTTPBasicAuth(username, password)
            )
            response.raise_for_status()
            print(f"Uploaded {arch}/repodata.json")


if __name__ == "__main__":
    dotenv_path = Path(__file__).parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    password = os.environ.get("WHEEL_SERVER_PASSWORD")
    parser = argparse.ArgumentParser(description="Upload repository to server")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    upload_parser = subparsers.add_parser("upload", help="Upload repository")
    upload_parser.add_argument("--host", help="Host URL")
    upload_parser.add_argument("--channel", help="Channel name")
    upload_parser.add_argument("--username", help="Username", default="admin")

    args = parser.parse_args()

    if args.command == "upload":
        host = args.host or os.environ.get("WHEEL_SERVER_HOST")
        channel = args.channel or os.environ.get("WHEEL_SERVER_CHANNEL")
        username = args.username or os.environ.get("WHEEL_SERVER_USERNAME", "admin")

        if not host:
            raise ValueError(
                "Host URL is not specified and WHEEL_SERVER_HOST environment variable is not set"
            )
        if not channel:
            raise ValueError(
                "Channel name is not specified and WHEEL_SERVER_CHANNEL environment variable is not set"
            )
        if not password:
            raise ValueError("WHEEL_SERVER_PASSWORD environment variable is not set")

        upload_repo(
            host=host,
            channel=channel,
            username=username,
            password=password,
        )
    else:
        parser.print_help()
