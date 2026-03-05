import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_random_baselines(meta_csv_path: Path) -> dict[str, float]:
	meta_df = pd.read_csv(meta_csv_path)
	return {
		str(row["Eval Task"]): float(row["Random baseline"])
		for _, row in meta_df.iterrows()
	}


def centered_to_uncentered(centered_score: float, random_baseline_percent: float) -> float:
	random_baseline = 0.01 * random_baseline_percent
	return centered_score * (1.0 - random_baseline) + random_baseline


def main() -> None:
	df = pd.read_csv("core_mcmc_sweep.csv")
	df = df.sort_values("mcmc_num_steps").drop_duplicates(subset=["mcmc_num_steps"], keep="last")
	meta_csv_path = Path("dclm/eval/eval_meta_data.csv")
	if not meta_csv_path.exists():
		raise FileNotFoundError(f"Could not find random baseline metadata at: {meta_csv_path}")
	random_baselines = load_random_baselines(meta_csv_path)

	steps = []
	summed_scores = []
	for _, row in df.iterrows():
		centered_task_scores = json.loads(row["centered_results_json"])
		uncentered_task_scores = []
		for task_name, centered_score in centered_task_scores.items():
			if task_name not in random_baselines:
				raise KeyError(f"Missing random baseline for task: {task_name}")
			uncentered_task_scores.append(
				centered_to_uncentered(float(centered_score), random_baselines[task_name])
			)
		steps.append(int(row["mcmc_num_steps"]))
		summed_scores.append(sum(uncentered_task_scores))

	colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(steps))]
	plt.figure(figsize=(8, 5))
	plt.bar([str(step) for step in steps], summed_scores, color=colors)
	plt.axhline(0.0, color="black", linewidth=0.8)
	plt.xlabel("MCMC steps")
	plt.ylabel("Sum of uncentered task accuracies")
	plt.title("Summed uncentered CORE task performance by MCMC steps")
	plt.tight_layout()
	plt.savefig("./core_task_sum_mcmc_bars.png", dpi=300)


if __name__ == "__main__":
	main()
