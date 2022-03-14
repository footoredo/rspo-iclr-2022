import os
import argparse
import json
import subprocess
from shutil import copyfile
from rspo.multi_agent.utils import get_timestamp, mkdir, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir")
    parser.add_argument("--load-step", type=int)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--num-processes", type=int)

    args = parser.parse_args()

    save_dir = os.path.join("/tmp/collect_trajectories", get_timestamp())
    mkdir(save_dir)

    config = json.load(open(os.path.join(args.load_dir, "config.json")))
    config.update(vars(args))

    episode_steps = config["episode_steps"]
    num_processes = args.num_processes
    num_episodes = args.num_episodes

    config["load"] = True
    config["num_env_steps"] = episode_steps * num_episodes
    config["num_steps"] = episode_steps * num_episodes // num_processes
    config["save_dir"] = save_dir
    config["play"] = False
    config["use_wandb"] = False
    config["train"] = False
    config["collect_trajectories"] = True
    config["no_load_refs"] = True

    config_path = os.path.join(save_dir, "config.json")
    json.dump(config, open(config_path, "w"))

    subprocess.run(["python", "ma_main.py", f"--config={config_path}"])

    env_config = config.get("env_config", None)
    env = make_env(config["env_name"], episode_steps, env_config)
    agents = env.agents

    for folder in os.listdir(save_dir):
        path = os.path.join(save_dir, folder, str(agents[0]), "trajectories.npy")
        if os.path.exists(path):
            for agent in agents:
                src = os.path.join(save_dir, folder, str(agent), "trajectories.npy")
                if args.load_step is not None:
                    dst = os.path.join(args.load_dir, str(agent), f"update-{args.load_step}", "trajectories.npy")
                else:
                    dst = os.path.join(args.load_dir, str(agent), "trajectories.npy")
                copyfile(src, dst)


if __name__ == "__main__":
    main()
