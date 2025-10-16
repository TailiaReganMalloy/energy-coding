"""Generate stylized W-shaped curve frames with a moving marker."""

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_w_curve(num_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
	"""Create a W-shaped curve with a taller left bump than the right."""

	x = np.linspace(-2.0, 2.0, num_points)
	y = ((x + 1.4) * (x + 0.35) * (x - 0.45) * (x - 1.2))
	return x, y


def locate_left_peak_and_minimum(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	"""Return the coordinates of the left peak and its adjacent minimum."""

	left_mask = x < 0
	left_indices = np.where(left_mask)[0]
	left_peak_index = left_indices[np.argmax(y[left_mask])]
	left_peak = (float(x[left_peak_index]), float(y[left_peak_index]))

	left_min_mask = (x > left_peak[0]) & (x < 0)
	if not np.any(left_min_mask):
		left_min_mask = x > left_peak[0]
	left_min_indices = np.where(left_min_mask)[0]
	left_min_index = left_min_indices[np.argmin(y[left_min_mask])]
	left_min = (float(x[left_min_index]), float(y[left_min_index]))

	return left_peak, left_min


def locate_right_peak_and_minimum(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	"""Return the coordinates of the right peak and its adjacent minimum."""

	right_mask = x > 0
	right_indices = np.where(right_mask)[0]
	right_peak_index = right_indices[np.argmax(y[right_mask])]
	right_peak = (float(x[right_peak_index]), float(y[right_peak_index]))

	right_min_mask = (x > 0) & (x < right_peak[0])
	if not np.any(right_min_mask):
		right_min_mask = x > 0
	right_min_indices = np.where(right_min_mask)[0]
	right_min_index = right_min_indices[np.argmin(y[right_min_mask])]
	right_min = (float(x[right_min_index]), float(y[right_min_index]))

	return right_peak, right_min


def plot_w_curve(
	ball_x: float,
	x: np.ndarray,
	y: np.ndarray,
	target_min: Tuple[float, float],
	include_arrow: bool = True,
	extra_highlights: Sequence[Tuple[float, float]] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
	"""Plot the W curve and place the marker at ``ball_x`` toward ``target_min``."""

	fig, ax = plt.subplots(figsize=(8, 4))
	sns.lineplot(x=x, y=y, ax=ax, color="#55CDFC", linewidth=10)

	ball_y = float(np.interp(ball_x, x, y))
	marker_kwargs = dict(color="black", s=500, zorder=5)
	ax.scatter(ball_x, ball_y, **marker_kwargs)

	highlight_points = {target_min}
	if extra_highlights:
		highlight_points.update(extra_highlights)

	for highlight_x, highlight_y in highlight_points:
		ax.scatter(highlight_x, highlight_y, color="#ff69b4", s=500, zorder=4)

	if include_arrow and not np.isclose(ball_x, target_min[0]):
		arrowprops = dict(arrowstyle="-|>", color="#ff69b4", linewidth=8)
		ax.annotate(
			"",
			xy=target_min,
			xytext=(ball_x, ball_y),
			arrowprops=arrowprops,
			zorder=6,
		)

	ax.axis("off")
	ax.set_facecolor("none")
	fig.patch.set_alpha(0)
	plt.tight_layout()
	return fig, ax


def create_w_curve_frames(output_dir: Path | str = "w_curve_frames") -> Sequence[Path]:
	"""Generate frames showing the marker moving toward both local minima."""

	sns.set_theme(style="white")
	x, y = generate_w_curve()
	left_peak, left_min = locate_left_peak_and_minimum(x, y)
	right_peak, right_min = locate_right_peak_and_minimum(x, y)
	distance = left_min[0] - left_peak[0]
	ball_positions = [
		left_peak[0],
		left_peak[0] + 0.45 * distance,
		left_peak[0] + 0.85 * distance,
		left_peak[0] + 1.15 * distance,
		left_min[0],
	]

	max_x = float(x[-1])
	right_distance = max_x - right_min[0]
	ball_positions_right = [
		max_x,
		right_min[0] + 0.75 * right_distance,
		right_min[0] + 0.5 * right_distance,
		right_min[0] + 0.25 * right_distance,
		right_min[0],
	]

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	frame_paths: List[Path] = []
	for idx, ball_x in enumerate(ball_positions, start=1):
		include_arrow = not np.isclose(ball_x, left_min[0])
		fig, _ = plot_w_curve(
			ball_x,
			x,
			y,
			left_min,
			include_arrow=include_arrow,
			extra_highlights=[right_min],
		)
		frame_path = output_path / f"w_curve_frame_{idx}.png"
		fig.savefig(frame_path, dpi=300, transparent=True)
		plt.close(fig)
		frame_paths.append(frame_path)

	starting_index = len(frame_paths) + 1
	for offset, ball_x in enumerate(ball_positions_right):
		include_arrow = not np.isclose(ball_x, right_min[0])
		fig, _ = plot_w_curve(
			ball_x,
			x,
			y,
			right_min,
			include_arrow=include_arrow,
		)
		frame_path = output_path / f"w_curve_frame_{starting_index + offset}.png"
		fig.savefig(frame_path, dpi=300, transparent=True)
		plt.close(fig)
		frame_paths.append(frame_path)

	return frame_paths


if __name__ == "__main__":
	paths = create_w_curve_frames()
	for path in paths:
		print(f"Saved {path}")

