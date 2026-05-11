import math

def compare_loss_lists(ref_losses, test_losses, tol=1e-3):
    if len(ref_losses) != len(test_losses):
        return False, f"Different lengths: {len(ref_losses)} vs {len(test_losses)}"
    for i, (a, b) in enumerate(zip(ref_losses, test_losses)):
        if abs(a - b) > tol:
            return False, f"Mismatch at step {i}: {a:.6f} vs {b:.6f}"
    return True, "Loss curves match within tolerance"

def perplexity_from_loss(loss):
    return math.exp(loss)