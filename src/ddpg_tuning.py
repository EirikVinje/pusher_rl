import os
import optuna
import pickle
from ddpg import Pusher



def objective(trial):

    memory = trial.suggest_categorical("memory", [10_000, 20_000, 30_000, 40_000, 50_000])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    steps = trial.suggest_categorical("steps", [100, 200, 300, 400, 500])

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    tau = trial.suggest_float("tau", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamme", 0.75, 0.95, log=True)
    
    epochs = int(EPOCHS/steps)

    pusher = Pusher(run_name="hp_run", 
                    epochs=epochs,
                    max_episode_steps=steps,
                    memory_size=memory,
                    batch_size=batch_size,
                    lr=lr,
                    tau=tau,
                    gamma=gamma,
                    record=False)
    
    reward = pusher.train()

    os.system(f"rm -rf /home/{os.environ['USER']}/data/pusher_models/hp_run")

    return reward


if __name__ == "__main__":
    
    name = "pusher_tuning1"

    study = optuna.create_study(study_name=name, direction="maximize")

    EPOCHS = 500_000
    
    study.optimize(objective, n_trials=1, show_progress_bar=False)
    
    print(study.best_params)
    print(study.best_value)

    pickle.dump(study, f"/home/{os.environ['USER']}/data/hp_res/{name}.pkl")
    