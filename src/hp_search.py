import os
import argparse
import optuna
from dqn import Pusher
from eval_model import Model

def objective(trial, args):
    # remove run_hp directory before running
    user = os.environ.get("USER")
    os.system(f"rm -rf /home/{user}/data/pusher_models/run_hp")
    seed = args.seed # which seed (track)
    device = args.device # cpu or cuda
    epochs = args.epochs # episodes
    save_n = args.save_n # how often to save
    run_name = args.run_name # name of run directory to save
    batch_size = args.batch_size # batch size

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    max_steps = trial.suggest_int("max_steps", 100, 1000)
    tau = trial.suggest_float("tau", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 1.0)
    memory_size = trial.suggest_int("memory_size", 1000, 10000)


    pusher = Pusher(seed=seed, 
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    run_name=run_name, 
                    save_n=save_n)
    
    pusher.train()

    model_str = f"{run_name}/{run_name}_{epochs}.pt"

    model = Model(model_str, render_mode='rgb_array', device=device)
    res = model.test()

    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--save_n", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default="run_hp")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=20)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

    