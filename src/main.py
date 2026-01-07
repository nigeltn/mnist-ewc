import argparse
import json
import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime

from mlp import MLP
from data import SplitMNIST
from engine import train_one_epoch, evaluate
import utils_ddp


def get_args():
    parser = argparse.ArgumentParser(description="Continual Learning: Split MNIST")

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="naive_baseline",
        help="Name identifier for the log file.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory where JSON results will be saved.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--epochs_per_task",
        type=int,
        default=5,
        help="Number of epochs to train on each specific task.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=400,
        help="Number of neurons in the hidden layers of the MLP.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cuda", "cpu"],
        help="Target device.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: uses small subset of data and runs on CPU.",
    )

    return parser.parse_args()


def save_json(data, log_dir, filename):
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {filepath}")


def main():
    args = get_args()

    if args.debug:
        print("💀 DEBUG MODE ENABLED: Skipping DDP setup.")
        device = torch.device("cpu")

        is_master = True
        world_size = 1
        args.epochs_per_task = 2
    else:
        local_rank, device, is_master = utils_ddp.setup_ddp()
        world_size = utils_ddp.get_world_size()

    utils_ddp.set_seed(args.seed)

    if is_master:
        print(f"--- Experiment: {args.experiment_name} ---")
        print(f"Device: {device}")
        print(f"World Size: {world_size}")

    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    dataset = SplitMNIST(root="./data", batch_size=args.batch_size)

    model = MLP(hidden_dim=args.hidden_dim).to(device)

    if not args.debug:
        model = DDP(model, device_ids=[device.index])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    experiment_log = {"config": vars(args), "results": []}

    for task_id, task_classes in enumerate(tasks):
        if is_master:
            print(f"\n[Task {task_id}] Training on classes {task_classes}...")

        temp_loader, _ = dataset.get_task_loader(task_classes)
        task_dataset = temp_loader.dataset

        if args.debug:
            train_loader = DataLoader(
                task_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
        else:
            sampler = DistributedSampler(task_dataset, shuffle=True)
            train_loader = DataLoader(
                task_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=2,
            )

        for epoch in range(args.epochs_per_task):
            if not args.debug:
                train_loader.sampler.set_epoch(epoch)

            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            if is_master and epoch % 5 == 0:
                print(f"   Epoch {epoch+1}/{args.epochs_per_task} | Loss: {loss:.4f}")

        if is_master:
            print(f"   Evaluation after Task {task_id}:")
            task_accuracies = []

            eval_model = model.module if hasattr(model, "module") else model

            for eval_task_id, eval_classes in enumerate(tasks):
                _, test_loader = dataset.get_task_loader(eval_classes)
                acc = evaluate(eval_model, test_loader, device)
                task_accuracies.append(acc)

                state = (
                    "CURRENT"
                    if eval_task_id == task_id
                    else ("PAST" if eval_task_id < task_id else "FUTURE")
                )
                print(f"      Task {eval_task_id} {eval_classes} [{state}]: {acc:.4f}")

            experiment_log["results"].append(
                {"training_task_id": task_id, "accuracies": task_accuracies}
            )

    if is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.experiment_name}_{timestamp}.json"
        save_json(experiment_log, args.log_dir, filename)

    if not args.debug:
        utils_ddp.cleanup_ddp()


if __name__ == "__main__":
    main()
