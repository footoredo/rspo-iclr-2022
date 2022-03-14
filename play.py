import os
import argparse
import json
import subprocess
from rspo.multi_agent.utils import get_timestamp, mkdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir")
    parser.add_argument("--new-ref-config")
    parser.add_argument("--load-step", type=int)
    parser.add_argument("--no-load-refs", action="store_true", default=False)
    parser.add_argument("--num-games-after-training", type=int, default=1)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", default=False)

    args = parser.parse_args()

    save_dir = os.path.join("/tmp/play", get_timestamp())
    mkdir(save_dir)

    config = json.load(open(os.path.join(args.load_dir, "config.json")))
    config.update(vars(args))

    config["load"] = True
    config["num_env_steps"] = 0
    config["num_processes"] = 1
    config["save_dir"] = save_dir
    config["play"] = True
    config["use_wandb"] = False

    num_refs = []
    if config["use_reference"]:
        ref_config = json.load(open(config["ref_config"]))
        for agent, _ref in ref_config.items():
            num_refs.append(len(_ref))
    else:
        for i in range(config["num_agents"]):
            num_refs.append(0)

    if args.new_ref_config is not None:
        config["ref_config"] = args.new_ref_config
        config["use_reference"] = True
        config["num_refs"] = num_refs

    config["load_dvd_weights_dir"] = None

    # print(config["num_refs"])
    # exit(0)

    config_path = os.path.join(save_dir, "config.json")
    json.dump(config, open(config_path, "w"))

    subprocess.run(["python", "ma_main.py", f"--config={config_path}"])

    if args.gif:
        import shutil
        shutil.move("/tmp/plays.gif", os.path.join(args.load_dir, "plays.gif"))


if __name__ == "__main__":
    main()
