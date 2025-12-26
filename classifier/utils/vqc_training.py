import os
import glob
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import jax
from jax import numpy as jnp
import optax
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from jaxopt import BFGS, LBFGS, ZoomLineSearch
import dill
import yaml

try:
    from utils.vqcs import LinearVQC, NonLinearVQC
except ImportError:
    from classifier.utils.vqcs import LinearVQC, NonLinearVQC


def _resolve_data_path(config):
    data_dir = config.get("data_dir")
    if data_dir:
        return os.path.join(os.fspath(data_dir), config["dataset_name"])
    warnings.warn(
        "config['data_dir'] not provided; falling back to basepath-derived data path.",
        RuntimeWarning,
    )
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(config["basepath"]))),
        "data",
        config["dataset_name"],
    )


def _load_dataset(config):
    try:
        path = _resolve_data_path(config)
        compression_depth = config.get("compression_depth", 0)
        if not compression_depth:
            labels = np.load(os.path.join(path, "labels.npy"))
            states = np.load(os.path.join(path, "states_p1.npy"))
            states = states[0]
        else:
            labels = np.load(os.path.join(path, "compressed/labels.npy"))
            states = np.load(os.path.join(path, f"compressed/states_p1_c{compression_depth}.npy"))
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            labels = labels[indices]
            states = states[indices]
    except FileNotFoundError as exc:
        raise ValueError("States file not found") from exc
    return states, labels

class Callback:
    def __init__(
        self,
        predict_fn,
        n_batches_train,
        n_batches_val,
        params,
        trial_dir,
        ):
        self.iteration = 0
        self.iteration_batch = {"train": 0, "val": 0}
        self.iteration_epoch = {"train": 0, "val": 0}
        self.n_batches = {"train": n_batches_train, "val": n_batches_val}
        self.trial_dir = trial_dir
        self.best_params = params

        self.lowest_val_loss = float("inf")
        self.losses = {"train": [], "val": []}
        self.accs = {"train": [], "val": []}
        self.losses_epoch = {"train": [], "val": []}
        self.accs_epoch = {"train": [], "val": []}
        self.writer = SummaryWriter(f"{trial_dir}")

        self.predict_fn = predict_fn

    def callback(
        self,
        params,
        batch,
        train_val: str,
        pbar=None
        ):
        states, targets = batch
        self.iteration_batch[train_val] += 1
        if isinstance(params, jax.Array):
            preds = jnp.asarray(self.predict_fn(params, states))
        else:
            preds = self.predict_fn(states).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
        loss = float(jnp.mean(optax.softmax_cross_entropy_with_integer_labels(preds, targets)))
        acc = float(jnp.mean(targets == jnp.argmax(preds, axis=1)))
        self.writer.add_scalar(f"metrics_batch/loss_{train_val}", loss, self.iteration_batch[train_val])
        self.writer.add_scalar(f"metrics_batch/accuracy_{train_val}", acc, self.iteration_batch[train_val])
        if train_val == "train" and pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                loss_train=str(loss)[:5],
                acc_train=str(acc)[:5]
                )

        if self.iteration_batch[train_val] % self.n_batches[train_val] == 0:
            self.iteration_epoch[train_val] += 1
            loss_epoch = float(jnp.mean(jnp.asarray(self.losses[train_val])))
            acc_epoch = float(jnp.mean(jnp.asarray(self.accs[train_val])))
            self.writer.add_scalar(f"metrics_epoch/loss_{train_val}", loss_epoch, self.iteration_epoch[train_val])
            self.writer.add_scalar(f"metrics_epoch/accuracy_{train_val}", acc_epoch, self.iteration_epoch[train_val])
            self.accs_epoch[train_val].append(acc_epoch)
            self.losses_epoch[train_val].append(loss_epoch)
            self.losses[train_val] = []
            self.accs[train_val] = []
            if train_val == "val" and loss_epoch < self.lowest_val_loss:
                self.lowest_val_loss = loss_epoch
                self.best_params = params
        else:
            self.losses[train_val].append(loss)
            self.accs[train_val].append(acc)


    def tb_to_pandas(self) -> None:
        event_file = glob.glob(os.path.join(self.trial_dir, 'events.out.tfevents.*'))[0]
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        tags = event_acc.Tags()
        data_batch, data_epoch = [], []
        tag_data_batch, tag_data_epoch = {}, {}

        for tag in tags['scalars']:
            if tag.startswith("metrics_batch"):
                data_batch = event_acc.Scalars(tag)
                tag_values = [d.value for d in data_batch]
                tag_data_batch[tag] = tag_values
            elif tag.startswith("metrics_epoch"):
                data_epoch = event_acc.Scalars(tag)
                tag_values = [d.value for d in data_epoch]
                tag_data_epoch[tag] = tag_values
            else:
                print(f"Skipping tag: {tag}")

        # data_batch = pd.DataFrame(tag_data_batch, index=[d.step for d in data_batch])
        data_epoch = pd.DataFrame(tag_data_epoch, index=[d.step for d in data_epoch])
        # data_batch.columns = data_batch.columns.str.replace('metrics_batch/', '')
        data_epoch.columns = data_epoch.columns.str.replace('metrics_epoch/', '')

        return data_epoch

class TrainingVQC:
    def __init__(
            self,
            # model,
            config):
        self.config = config

    def get_solver(self, optimizer_name, loss_fn, has_aux):
        optimizer_key = optimizer_name.lower()
        if optimizer_key == "bfgs":
            self.batch_iters = 10
            return self.get_bfgs(loss_fn, has_aux)
        elif optimizer_key == "lbfgs":
            self.batch_iters = 1
            return self.get_lbfgs(loss_fn, has_aux)
        elif optimizer_key == "adam":
            self.batch_iters = 1
            return self.get_adam(self.config["learning_rate"])
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    @staticmethod
    def get_bfgs(loss_fn, has_aux):
        zoom_line_search = ZoomLineSearch(loss_fn, verbose=0)
        solver = BFGS(
            fun=loss_fn,
            has_aux=has_aux,
            verbose=0,
            linesearch=zoom_line_search,
            tol=0
        )
        return solver

    @staticmethod
    def get_lbfgs(loss_fn, has_aux):
        solver = LBFGS(
            fun=loss_fn,
            maxiter=20,
            has_aux=has_aux,
            history_size=10,
        )
        return solver

    @staticmethod
    def get_adam(learning_rate):
        optimizer = optax.adam(learning_rate=learning_rate)
        return optimizer

    def train(self):
        seed = self.config.get("seed")
        if seed is not None:
            np.random.seed(seed)
            jax.random.PRNGKey(seed)

        states, labels = _load_dataset(self.config)

        if not self.config.get("n_qubits"):
            dim = states.shape[1]
            n_qubits = int(np.log2(dim))
            if 2**n_qubits != dim:
                raise ValueError(f"State dimension {dim} is not a power of two; cannot infer n_qubits.")
            self.config["n_qubits"] = n_qubits

        if self.config["model_name"] == "LinearVQC":
            model = LinearVQC(
                N_QUBITS=self.config["n_qubits"],
                DEPTH=self.config["depth"],
                building_block_tag=self.config["building_block_tag"],
                temperature=self.config["temperature"]).setup()
        elif self.config["model_name"] == "NonLinearVQC":
            model = NonLinearVQC(
                N_QUBITS=self.config["n_qubits"],
                DEPTH=self.config["depth"],
                use_initial_state=False,
                building_block_tag=self.config["building_block_tag"],
                temperature=self.config["temperature"]).setup()
        elif self.config["model_name"] == "NonLinearVQC_shadow":
            model = NonLinearVQC(
                N_QUBITS=self.config["n_qubits"],
                DEPTH=self.config["depth"],
                use_initial_state=True,
                building_block_tag=self.config["building_block_tag"],
                temperature=self.config["temperature"]).setup()
        else:
            raise ValueError(f"Unknown model: {self.config['model_name']}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        splits = list(skf.split(states, labels))
        train_idx, val_idx = splits[self.config["fold"]]

        states_train, targets_train = states[train_idx], labels[train_idx]
        states_val, targets_val = states[val_idx], labels[val_idx]
        n_batches = max(1, len(states_train) // self.config["batch_size"])

        states_train_batches = np.array_split(states_train, n_batches)
        states_val_batches = np.array_split(states_val, n_batches)
        targets_train_batches = np.array_split(targets_train, n_batches)
        targets_val_batches = np.array_split(targets_val, n_batches)

        params = model["params"]

        loss_fn = lambda params, *args: jnp.mean(model["loss_fn"](params, *args))
        predict_fn = model["model_vmap"]
        grad_fn_vmap = model["grad_fn"]

        predict_fn = lambda params, input: model["model_vmap"](params, input)

        cb = Callback(
            predict_fn=predict_fn,
            n_batches_train=len(states_train_batches),
            n_batches_val=len(states_val_batches),
            params=params,
            trial_dir=self.config["trial_dir"]
        )

        optimizer_key = self.config["optimizer"].lower()
        if optimizer_key == "adam":
            solver = self.get_solver(self.config["optimizer"], loss_fn, has_aux=False)
            solver_state = solver.init(params)
        else:
            loss_fn_mean = lambda params, state, target: jnp.mean(loss_fn(params, state, target))
            solver = self.get_solver(self.config["optimizer"], loss_fn_mean, has_aux=False)

        params_epoch = []
        best_val_loss = float("inf")
        best_val_acc = None
        best_epoch = None
        train_acc_at_best = None
        patience_raw = self.config.get("early_stopping_patience", 10)
        patience = int(patience_raw) if patience_raw is not None else 0
        min_delta = float(self.config.get("min_delta", 0.0))
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            with tqdm(total=len(states_train_batches), leave=False) as pbar:
                pbar.set_description(f"Epoch {epoch + 1}/{self.config['epochs']}")
                for batch in zip(states_train_batches, targets_train_batches):
                    if optimizer_key != "adam":
                        state = solver.init_state(params, *batch)
                    for _ in range(self.batch_iters):

                        if optimizer_key == "adam":
                            gradient = jnp.mean(grad_fn_vmap(params, *batch), axis=0)
                            updates, solver_state = solver.update(gradient, solver_state, params)
                            params = optax.apply_updates(params, updates)
                        else:
                            params, state = solver.update(params, state, *batch)

                    cb.callback(params, batch, "train", pbar)
            params_epoch.append(np.asarray(params))
            pbar.close()
            for batch in zip(states_val_batches, targets_val_batches):
                cb.callback(params, batch, "val")

            if cb.losses_epoch["val"]:
                val_loss = cb.losses_epoch["val"][-1]
                val_acc = cb.accs_epoch["val"][-1]
                train_acc = cb.accs_epoch["train"][-1] if cb.accs_epoch["train"] else None
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    train_acc_at_best = train_acc
                    cb.best_params = params
                    cb.lowest_val_loss = best_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience > 0 and patience_counter >= patience:
                        break
        cb.writer.close()

        # Move the saving stuff to handling function?
        data_epoch = cb.tb_to_pandas()
        # data_batch.to_csv(f"{self.config['trial_dir']}/training_data_batch.csv", index=False)
        data_epoch.to_csv(f"{self.config['trial_dir']}/training_data_epoch.csv", index=False)

        if "predict_fn" in model:
            predict_fn = lambda params, input: model["predict_fn"](params, model["batch_stats"], input)
            with open(f"{self.config['trial_dir']}/predict_fn.pkl", 'wb') as f:
                dill.dump(predict_fn, f)

        with open(f"{self.config['trial_dir']}/params_best.pkl", 'wb') as f:
            dill.dump(cb.best_params, f)

        with open(f"{self.config['trial_dir']}/params_epoch.pkl", 'wb') as f:
            dill.dump(params_epoch, f)

        if best_epoch is None and cb.losses_epoch["val"]:
            best_epoch = len(cb.losses_epoch["val"])
            best_val_loss = cb.losses_epoch["val"][-1]
            best_val_acc = cb.accs_epoch["val"][-1]
            train_acc_at_best = cb.accs_epoch["train"][-1] if cb.accs_epoch["train"] else None

        n_params = int(np.asarray(params).size)
        summary = {
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "train_acc_at_best": train_acc_at_best,
            "n_params": n_params,
        }

        return predict_fn, cb.best_params, summary

def main(config_model, use_ray=True):
    if use_ray:
        from ray.air import session
        trial_dir = session.get_trial_dir()
        config_model["trial_dir"] = trial_dir
    else:
        config_model["trial_dir"] = config_model["basepath"]

    os.makedirs(config_model["trial_dir"], exist_ok=True)
    with open(os.path.join(config_model["trial_dir"], "config.yaml"), 'w') as f:
        yaml.dump(config_model, f)

    training = TrainingVQC(config_model)
    predict_fn, best_params, summary = training.train()
    return summary

if __name__ == "__main__":
    from datetime import datetime

    basepath = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{basepath}/results_ray_tmp/{timestamp}"

    config = {
        "basepath": result_dir,
        "learning_rate": 8e-4,
        "epochs": 100,
        "n_qubits": 11,
        "depth": 4,
        "building_block_tag": "su4",
        "temperature": 128,
        "optimizer": "adam",
        "model_name": "NonLinearVQC_shadow", # "LinearVQC", "NonLinearVQC", "NonLinearVQC_shadow"
        "dataset_name": "mnist",
        "fold": 0,
        "compression_depth": 1,
        "batch_size": 100,
    }

    main(config, use_ray=False)
