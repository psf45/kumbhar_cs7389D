import os
import time
import argparse
import torch
from model_gpt import TinyGPT
from data import SyntheticTokenDataset, build_loader
from perf import RegionTimer, append_csv
from dist import init_distributed, cleanup_distributed, is_main_process


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=5000)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--results_csv", type=str, default="results/baseline_results.csv")
    p.add_argument("--simulate_io", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    info = init_distributed()

    distributed = info["distributed"]
    rank = info["rank"]
    world_size = info["world_size"]
    local_rank = info["local_rank"]
    hostname = info["hostname"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        gpu_name = torch.cuda.get_device_name(local_rank)
    else:
        gpu_name = "cpu"

    os.makedirs("results", exist_ok=True)

    dataset = SyntheticTokenDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=1234
    )

    loader, sampler = build_loader(
        dataset,
        batch_size=args.batch_size,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )

    model = TinyGPT(
        vocab_size=args.vocab_size,
        block_size=args.seq_len,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device == "cuda" else None
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    timer = RegionTimer()
    step_losses = []

    total_start = time.perf_counter()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step, (x, y) in enumerate(loader):
            if args.simulate_io:
                time.sleep(0.002)
            timer.start("data_to_device")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            timer.stop("data_to_device")

            optimizer.zero_grad(set_to_none=True)

            timer.start("forward")
            logits, loss = model(x, y)
            timer.stop("forward")

            timer.start("backward")
            loss.backward()
            timer.stop("backward")

            timer.start("optimizer_step")
            optimizer.step()
            timer.stop("optimizer_step")

            loss_val = loss.item()
            epoch_loss += loss_val
            step_count += 1

            if is_main_process() and step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss_val:.6f}")

            if is_main_process():
                step_losses.append(loss_val)

        avg_loss = epoch_loss / max(step_count, 1)
        if is_main_process():
            print(f"epoch={epoch} avg_loss={avg_loss:.6f}")

    total_time = time.perf_counter() - total_start

    if is_main_process():
        row = {
            "hostname": hostname,
            "device": device,
            "gpu_name": gpu_name,
            "distributed": distributed,
            "world_size": world_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_samples": args.num_samples,
            "runtime_sec": round(total_time, 4),
            "forward_sec": round(timer.snapshot().get("forward", 0.0), 4),
            "backward_sec": round(timer.snapshot().get("backward", 0.0), 4),
            "optimizer_step_sec": round(timer.snapshot().get("optimizer_step", 0.0), 4),
            "data_to_device_sec": round(timer.snapshot().get("data_to_device", 0.0), 4),
            "final_loss": round(step_losses[-1], 6) if step_losses else None,
        }
        append_csv(args.results_csv, row)
        print("\nTraining complete")
        print(f"Total runtime: {total_time:.4f} sec")
        print(f"Results saved to: {args.results_csv}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
