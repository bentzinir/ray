import os
from pandas_ods_reader import read_ods
import argparse


def main(args):
    sheet_idx = 1
    df = read_ods(args.servers, sheet_idx)

    for index, row in df.iterrows():
        ip = row["IP"]
        username = row["username"]
        remote_dir = row["directory"]
        dir = os.path.join(args.local_dir, ip)
        if not os.path.exists(dir):
            os.makedirs(dir)
        buffer = f"sshfs {username}@{ip}:{remote_dir} {args.local_dir}/{ip}"
        # print(buffer)
        os.system(buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="none")
    parser.add_argument("--servers", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    assert args.servers != 'none'
    assert args.local_dir != 'none'
    main(args)
