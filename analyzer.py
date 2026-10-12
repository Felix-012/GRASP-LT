from collections import defaultdict, Counter
from typing import Sequence, Iterable, Optional
from logging import Logger
from pathlib import Path
from datetime import datetime
import torch

class ExpertAnalyzer:
    def __init__(self, num_experts: int, logger: Logger, current_step: int = 0,
                 report_step_interval: int = 500, report_path: Optional[Path] = None,
                 run_id: Optional[str] = None):
        self.num_experts = num_experts
        self.logger = logger
        self.current_step = current_step
        self.last_report_step = current_step
        self.report_step_interval = report_step_interval
        self.total_num_samples = [0] * num_experts
        self.labels_per_expert = defaultdict(list)
        self.report_path = Path(report_path).expanduser() if report_path else None
        self.run_id = run_id or "unknown"

    def _build_report_string(self, current_step: int) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append("----- Current Expert Report -----")
        lines.append(f"Run ID: {self.run_id}")
        lines.append(f"Generated at: {now_str}")
        lines.append(f"Current Step: {current_step}")

        total_seen = sum(self.total_num_samples)
        for expert in range(self.num_experts):
            exp_total = self.total_num_samples[expert]
            share = (exp_total / total_seen) if total_seen else 0.0
            lines.append(f"Expert {expert}:")
            lines.append(f"Share of samples: {share:.4f}")
            if exp_total:
                counts = Counter(self.labels_per_expert[expert])
                count_lines = [f"{s}: {n / exp_total * 100:.2f}%" for s, n in counts.most_common()]
                count_string = "\n".join(count_lines) if count_lines else "(no labels)"
            else:
                count_string = "(no samples yet)"
            lines.append(f"Label distribution:\n{count_string}")

        return "\n".join(lines)

    def _write_report(self, report_text: str):
        if not self.report_path:
            return
        try:
            if self.report_path.parent and not self.report_path.parent.exists():
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
            # Overwrite so only the latest report for this run is saved
            with self.report_path.open("w", encoding="utf-8") as f:
                f.write(report_text + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write report to '{self.report_path}': {e}")

    def update(self, expert_assignments: Sequence[int], current_step: int, labels: Iterable[str]):
        if isinstance(expert_assignments, torch.Tensor):
            expert_assignments = expert_assignments.tolist()
        for expert, label_string in zip(expert_assignments, labels):
            if not (0 <= expert < self.num_experts):
                self.logger.warning(f"Ignoring out-of-range expert index {expert}")
                continue
            self.total_num_samples[expert] += 1
            if isinstance(label_string, (list, tuple, set)):
                labs = [str(x).strip() for x in label_string if str(x).strip()]
            else:
                labs = [s.strip() for s in str(label_string).split("|") if s.strip()]
            self.labels_per_expert[expert].extend(labs)

        if (current_step - self.last_report_step) >= self.report_step_interval:
            # Log as before
            self.logger.info("----- Current Expert Report -----")
            self.logger.info(f"Run ID: {self.run_id}")
            self.logger.info(f"Current Step: {current_step}")

            total_seen = sum(self.total_num_samples)
            for expert in range(self.num_experts):
                exp_total = self.total_num_samples[expert]
                share = (exp_total / total_seen) if total_seen else 0.0
                self.logger.info(f"Expert {expert}:")
                self.logger.info(f"Share of samples: {share:.4f}")
                if exp_total:
                    counts = Counter(self.labels_per_expert[expert])
                    count_lines = [f"{s}: {n / exp_total * 100:.2f}%" for s, n in counts.most_common()]
                    count_string = "\n".join(count_lines) if count_lines else "(no labels)"
                else:
                    count_string = "(no samples yet)"
                self.logger.info(f"Label distribution:\n{count_string}")

            # Build + overwrite file so only latest timestamp for this run remains
            report_text = self._build_report_string(current_step)
            self._write_report(report_text)

            self.last_report_step = current_step

        self.current_step = current_step
