from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from src.evaluation.metrics import GeneralizationResult, MetricsResult


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Arabic IR Pipeline Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 32px; }}
  h3 {{ color: #3949ab; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
  .config-item {{ background: #e8eaf6; border-radius: 6px; padding: 10px 14px; }}
  .config-key {{ font-weight: bold; color: #1a237e; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }}
  .config-val {{ color: #333; margin-top: 2px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.93em; }}
  th {{ background: #1a237e; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #e0e0e0; }}
  tr:hover {{ background: #e8eaf6; }}
  tr:nth-child(even) {{ background: #f5f5ff; }}
  .best {{ font-weight: bold; color: #1b5e20; }}
  .delta-pos {{ color: #2e7d32; font-weight: bold; }}
  .delta-neg {{ color: #c62828; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px; }}
  .badge-kd {{ background: #c8e6c9; color: #1b5e20; }}
  .badge-nokd {{ background: #ffcdd2; color: #b71c1c; }}
  .badge-lora {{ background: #e3f2fd; color: #0d47a1; }}
  .badge-hn {{ background: #fff9c4; color: #f57f17; }}
  .badge-mat {{ background: #f3e5f5; color: #4a148c; }}
  .gr-high  {{ color: #1b5e20; font-weight: bold; }}
  .gr-med   {{ color: #e65100; font-weight: bold; }}
  .gr-low   {{ color: #b71c1c; font-weight: bold; }}
  .chart-container {{ text-align: center; margin: 16px 0; }}
  img {{ max-width: 100%; border-radius: 6px; }}
  .footer {{ text-align: center; color: #9e9e9e; font-size: 0.85em; margin-top: 40px; padding-top: 16px; border-top: 1px solid #e0e0e0; }}
  .summary-box {{ background: #e8f5e9; border-left: 4px solid #43a047; padding: 14px 18px; border-radius: 0 6px 6px 0; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Arabic IR Pipeline Report</h1>
<p style="color:#666;">Generated: {timestamp}</p>

{config_section}

{summary_section}

{metrics_section}

{comparison_section}

{gr_section}

{charts_section}

<div class="footer">Arabic IR &amp; Reranking with Knowledge Distillation — auto-generated report</div>
</body>
</html>
"""


class ReportGenerator:
    """
    Generates an HTML report from a list of MetricsResult objects and
    the pipeline configuration.
    """

    def __init__(
        self,
        config,
        results: Optional[List[MetricsResult]] = None,
        gr_results: Optional[List[GeneralizationResult]] = None,
    ):
        self.config = config
        self.results: List[MetricsResult] = results or []
        self.gr_results: List[GeneralizationResult] = gr_results or []
        self._chart_paths: List[str] = []

    def add_result(self, result: MetricsResult):
        self.results.append(result)

    def add_gr_result(self, gr: GeneralizationResult):
        self.gr_results.append(gr)

    def add_chart(self, path: str):
        """Register an externally-generated chart image to embed."""
        self._chart_paths.append(path)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, output_path: Optional[str] = None) -> str:
        out_path = output_path or self.config.report_output
        html = _HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_section=self._render_config(),
            summary_section=self._render_summary(),
            metrics_section=self._render_metrics_table(),
            comparison_section=self._render_comparison(),
            gr_section=self._render_gr_section(),
            charts_section=self._render_charts(),
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _render_config(self) -> str:
        cfg = self.config
        items = [
            ("Dataset", cfg.data.dataset),
            ("Base model", f"{cfg.model.base_model} ({cfg.model.get_hf_model_id()})"),
            ("Encoder type", cfg.model.encoder_type),
            ("Knowledge distillation", f"{cfg.training.use_kd} | mode={cfg.training.kd_mode}"),
            ("KD α (label smooth)", cfg.training.kd_alpha),
            ("KD temperature", cfg.training.kd_temperature),
            ("Hard negatives", f"{cfg.training.hard_negatives} | {cfg.training.hn_strategy}"),
            ("LoRA", cfg.model.use_lora),
            ("Matryoshka", f"{cfg.model.use_matryoshka} | dims={cfg.model.matryoshka_dims}"),
            ("MNRL hybrid", f"{cfg.training.use_mnrl_hybrid} | w={cfg.training.mnrl_weight}"),
            ("Epochs", cfg.training.num_train_epochs),
            ("Batch size", f"{cfg.training.per_device_train_batch_size} × accum {cfg.training.gradient_accumulation_steps}"),
            ("Learning rate", cfg.training.learning_rate),
            ("Seed", cfg.seed),
        ]
        cards = "\n".join(
            f'<div class="config-item"><div class="config-key">{k}</div><div class="config-val">{v}</div></div>'
            for k, v in items
        )
        return f'<div class="card"><h2>Configuration</h2><div class="config-grid">{cards}</div></div>'

    def _render_summary(self) -> str:
        if not self.results:
            return ""
        best_mrr = max((r.mrr_at_10 for r in self.results), default=0.0)
        best_recall = max((r.recall_at_1000 for r in self.results), default=0.0)
        best_ndcg = max((r.ndcg_at_10 for r in self.results), default=0.0)
        n_models = len(set(r.model_name for r in self.results))
        box = (
            f"<b>{n_models}</b> model configurations evaluated. "
            f"Best MRR@10: <b>{best_mrr:.4f}</b> &nbsp;|&nbsp; "
            f"Best NDCG@10: <b>{best_ndcg:.4f}</b> &nbsp;|&nbsp; "
            f"Best R@1000: <b>{best_recall:.4f}</b>"
        )
        return f'<div class="card"><h2>Summary</h2><div class="summary-box">{box}</div></div>'

    def _render_metrics_table(self) -> str:
        if not self.results:
            return ""

        rows_data = [r.as_dict() for r in self.results]

        # Find column-wise best values for highlighting
        num_cols = ["MRR@10", "NDCG@10", "MAP@10", "R@10", "R@100", "R@1000"]
        best_vals = {col: max((r.get(col, 0.0) for r in rows_data), default=0.0) for col in num_cols}

        header_cols = ["model", "stage", "dataset", "kd_mode", "lora", "hard_neg", "matryoshka"] + num_cols
        header_html = "".join(f"<th>{c}</th>" for c in header_cols)

        rows_html = ""
        for row in rows_data:
            cells = ""
            for col in header_cols:
                val = row.get(col, "")
                if col in num_cols and isinstance(val, float):
                    css = ' class="best"' if abs(val - best_vals[col]) < 1e-6 else ""
                    cells += f"<td{css}>{val:.4f}</td>"
                elif col in ("lora", "hard_neg", "matryoshka"):
                    badge_cls = "badge-lora" if col == "lora" else ("badge-hn" if col == "hard_neg" else "badge-mat")
                    label = "✓" if val else "✗"
                    color = badge_cls if val else ""
                    cells += f'<td><span class="badge {color}">{label}</span></td>'
                elif col == "kd_mode":
                    badge = "badge-kd" if "kd" in str(val).lower() else "badge-nokd"
                    cells += f'<td><span class="badge {badge}">{val}</span></td>'
                else:
                    cells += f"<td>{val}</td>"
            rows_html += f"<tr>{cells}</tr>"

        table = f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"
        return f'<div class="card"><h2>Evaluation Metrics</h2>{table}</div>'

    def _render_comparison(self) -> str:
        """KD vs NoKD delta table."""
        kd_results = [r for r in self.results if "kd" in r.kd_mode.lower()]
        nokd_results = [r for r in self.results if r.kd_mode == "nokd"]
        if not kd_results or not nokd_results:
            return ""

        rows = ""
        for kd_r in kd_results:
            # Find matching NoKD result (same model base, same stage)
            match = next(
                (r for r in nokd_results if r.stage == kd_r.stage and r.dataset == kd_r.dataset),
                None,
            )
            if match is None:
                continue
            for metric, kd_val, nokd_val in [
                ("MRR@10", kd_r.mrr_at_10, match.mrr_at_10),
                ("NDCG@10", kd_r.ndcg_at_10, match.ndcg_at_10),
                ("R@1000", kd_r.recall_at_1000, match.recall_at_1000),
            ]:
                delta = kd_val - nokd_val
                rel = (delta / nokd_val * 100) if nokd_val > 0 else 0.0
                sign = "+" if delta >= 0 else ""
                css = "delta-pos" if delta >= 0 else "delta-neg"
                rows += (
                    f"<tr><td>{kd_r.model_name}</td><td>{kd_r.stage}</td>"
                    f"<td>{metric}</td><td>{nokd_val:.4f}</td><td>{kd_val:.4f}</td>"
                    f'<td class="{css}">{sign}{delta:.4f} ({sign}{rel:.1f}%)</td></tr>'
                )

        if not rows:
            return ""

        table = (
            "<table><thead><tr>"
            "<th>Model</th><th>Stage</th><th>Metric</th>"
            "<th>NoKD</th><th>KD</th><th>Delta (relative)</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )
        return f'<div class="card"><h2>KD vs NoKD Comparison</h2>{table}</div>'

    def _render_gr_section(self) -> str:
        """
        Generalization Ratio table.

        Columns: model | stage | kd_mode | in-domain dataset | zero-shot dataset |
                 in-domain metrics | zero-shot metrics | GR values

        GR colour coding:
          ≥ 1.0  → green  (fully generalises)
          0.8–1  → orange (partial generalisation)
          < 0.8  → red    (poor generalisation)
        """
        if not self.gr_results:
            return ""

        # Collect GR metric names from first result
        gr_keys = [k for k in self.gr_results[0].gr.keys()]
        base_metrics = [k[3:] for k in gr_keys]   # strip "GR_" prefix

        header_cols = (
            ["model", "stage", "kd_mode", "in_domain", "zero_shot"]
            + [f"in {m}" for m in base_metrics]
            + [f"zs {m}" for m in base_metrics]
            + gr_keys
        )
        header_html = "".join(f"<th>{c}</th>" for c in header_cols)

        rows_html = ""
        for gr in self.gr_results:
            cells = (
                f"<td>{gr.model_name}</td>"
                f"<td>{gr.stage}</td>"
                f"<td>{gr.kd_mode}</td>"
                f"<td>{gr.in_domain.dataset}</td>"
                f"<td>{gr.zero_shot.dataset}</td>"
            )
            for m in base_metrics:
                cells += f"<td>{gr.in_domain.get(m):.4f}</td>"
            for m in base_metrics:
                cells += f"<td>{gr.zero_shot.get(m):.4f}</td>"
            for key in gr_keys:
                val = gr.gr.get(key, 0.0)
                if val >= 1.0:
                    css = "gr-high"
                elif val >= 0.8:
                    css = "gr-med"
                else:
                    css = "gr-low"
                cells += f'<td class="{css}">{val:.4f}</td>'
            rows_html += f"<tr>{cells}</tr>"

        table = (
            f"<table><thead><tr>{header_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

        legend = (
            '<p style="font-size:0.88em; color:#555; margin-top:10px;">'
            '<span class="gr-high">■</span> GR ≥ 1.0 — fully generalises &nbsp;'
            '<span class="gr-med">■</span> GR 0.8–1.0 — partial generalisation &nbsp;'
            '<span class="gr-low">■</span> GR &lt; 0.8 — poor generalisation</p>'
        )

        note = (
            "<p style='font-size:0.88em;color:#666;'>"
            "GR = zero-shot metric ÷ in-domain metric. "
            "Bi-encoder evaluated on FAISS top-1000 candidates; "
            "Cross-encoder evaluated on BM25 candidates.</p>"
        )

        return f'<div class="card"><h2>Generalization Ratio (GR)</h2>{note}{table}{legend}</div>'

    def _render_charts(self) -> str:
        if not self._chart_paths:
            return ""
        imgs = "".join(
            f'<div class="chart-container"><img src="{p}" alt="{os.path.basename(p)}"/></div>'
            for p in self._chart_paths
        )
        return f'<div class="card"><h2>Charts</h2>{imgs}</div>'

    # ------------------------------------------------------------------
    # Chart helpers (call externally, then add_chart())
    # ------------------------------------------------------------------

    def plot_metrics_bar(
        self,
        output_path: str = "charts/metrics_bar.png",
        metric: str = "MRR@10",
    ) -> str:
        """Generate a bar chart comparing metric across all results."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping chart")
            return ""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        labels = [f"{r.model_name}\n({r.stage})" for r in self.results]
        values = [getattr(r, metric.lower().replace("@", "_at_").replace("r_at", "recall_at"), 0.0) for r in self.results]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
        colors = ["#42a5f5" if "kd" in r.kd_mode else "#ef5350" for r in self.results]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"{metric} Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.add_chart(output_path)
        return output_path

    def plot_gr_bar(
        self,
        output_path: str = "charts/gr_bar.png",
        metric: str = "GR_MRR@10",
    ) -> str:
        """Bar chart comparing GR across all GeneralizationResult objects."""
        if not self.gr_results:
            return ""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return ""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        labels = [f"{g.model_name}\n({g.stage})" for g in self.gr_results]
        values = [g.gr.get(metric, 0.0) for g in self.gr_results]

        colors = ["#66bb6a" if v >= 1.0 else ("#ffa726" if v >= 0.8 else "#ef5350") for v in values]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(1.0, color="#333", linewidth=1.2, linestyle="--", label="GR = 1.0 (full generalisation)")
        ax.axhline(0.8, color="#e65100", linewidth=0.8, linestyle=":", alpha=0.7, label="GR = 0.8 (threshold)")
        ax.set_title(f"{metric} — Generalization Ratio", fontsize=13, fontweight="bold")
        ax.set_ylabel("GR (zero-shot / in-domain)")
        ax.set_ylim(0, max(max(values) * 1.15, 1.2))
        ax.legend(fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.add_chart(output_path)
        return output_path

    def plot_training_loss(
        self,
        loss_history: List[float],
        output_path: str = "charts/training_loss.png",
        label: str = "Training Loss",
    ) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return ""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(loss_history, color="#1a237e", linewidth=1.5)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.add_chart(output_path)
        return output_path
