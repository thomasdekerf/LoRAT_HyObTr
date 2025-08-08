import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the tracker on HOT")
    parser.add_argument("method_name", type=str, help="Method name under config/")
    parser.add_argument("config_name", type=str, help="Config file name inside config/{method_name}/")
    parser.add_argument("data_root", type=str, help="Path to HOT dataset root")
    parser.add_argument("pretrained", type=str, help="Path to pretrained weight file")
    parser.add_argument("output_dir", type=str, help="Directory to store outputs")
    args, unknown = parser.parse_known_args()

    # Expose dataset location to the configuration system
    os.environ["HOT_PATH"] = args.data_root

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "main.py"),
        args.method_name,
        args.config_name,
        "--output_dir",
        args.output_dir,
        "--weight_path",
        args.pretrained,
    ] + unknown

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
