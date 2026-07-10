import json
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Union

from src.datasets.bipia_loader import BIPIALoader
from src.datasets.injection import inject
from src.core.constants import SEED, SOURCE_BIPIA, SCHEMA_COLUMNS
from src.core.enums import Label, Split
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """Builds the binary classification dataset from BIPIA resources."""

    def __init__(
        self,
        task: str = "email",
        positions: List[str] = None,
        balance: bool = False,
        seed: int = SEED,
        bipia_root: str = "data/raw/bipia",
        target_attack_samples: int = None,
    ):
        self.task = task
        self.positions = positions or ["start", "middle", "end"]
        self.balance = balance
        self.seed = seed
        self.target_attack_samples = target_attack_samples
        
        self.loader = BIPIALoader(bipia_root=bipia_root)
        self.output_records = []

    def build(self, splits: List[str] = None, generated_benign_path: str = None) -> pd.DataFrame:
        """Construct the dataset for the requested splits.

        Args:
            splits: List of splits to process (default: ["train", "test"]).
            generated_benign_path: Path to the JSONL containing generated benign intents.

        Returns:
            Unified pandas DataFrame conforming to SCHEMA_COLUMNS.
        """
        if splits is None:
            splits = [Split.TRAIN.value, Split.TEST.value]

        # 1. Load Generated Benign Intents (if provided)
        from collections import defaultdict
        generated_intents_by_context = defaultdict(list)
        if generated_benign_path:
            logger.info(f"Loading generated benign intents from {generated_benign_path}")
            from src.utils.io import read_jsonl
            generated_data = read_jsonl(generated_benign_path)
            for item in generated_data:
                generated_intents_by_context[item["context"]].append(item.get("user_intent", ""))
        
        for split in splits:
            logger.info(f"Building dataset for split '{split}'...")
            
            contexts = self.loader.load_contexts(self.task, split)
            attacks = self.loader.load_attack_library(split)
            
            if not contexts:
                continue

            # 2. Add Benign samples
            if generated_benign_path:
                for ctx_idx, ctx in enumerate(contexts):
                    actual_split = split
                    if split == Split.TRAIN.value and ctx_idx % 5 == 0:
                        actual_split = Split.VALIDATION.value
                        
                    raw_text = ctx.get("context", ctx.get("text", ""))
                    
                    # Safely map all generated intents directly to their original context's split
                    gen_intents = generated_intents_by_context.get(raw_text, [])
                    for i, intent in enumerate(gen_intents):
                        self.output_records.append({
                            "id": f"{SOURCE_BIPIA}_{actual_split}_gen_benign_{ctx_idx}_{i}",
                            "context": raw_text,
                            "attack_instruction": intent,
                            "label": Label.BENIGN.value,
                            "attack_family": "benign",
                            "attack_position": None,
                            "split": actual_split,
                            "task": self.task,
                            "source": "generated",
                        })
            else:
                for ctx_idx, ctx in enumerate(contexts):
                    actual_split = split
                    if split == Split.TRAIN.value and ctx_idx % 5 == 0:
                        actual_split = Split.VALIDATION.value
                        
                    raw_text = ctx.get("context", ctx.get("text", ""))
                    
                    self.output_records.append({
                        "id": f"{SOURCE_BIPIA}_{actual_split}_benign_{ctx_idx}",
                        "context": raw_text,
                        "attack_instruction": None,
                        "label": Label.BENIGN.value,
                        "attack_family": "benign",
                        "attack_position": None,
                        "split": actual_split,
                        "task": self.task,
                        "source": SOURCE_BIPIA,
                    })

            # 3. Add Attack samples (label = 1)
            # Combinatorial expansion: every context * every payload * every position
            attack_idx = 0
            for ctx_idx, ctx in enumerate(contexts):
                actual_split = split
                if split == Split.TRAIN.value and ctx_idx % 5 == 0:
                    actual_split = Split.VALIDATION.value
                    
                raw_text = ctx.get("context", ctx.get("text", ""))
                
                for family, payloads in attacks.items():
                    for payload in payloads:
                        for pos in self.positions:
                            # Use official BIPIA injection logic
                            poisoned_text = inject(
                                context=raw_text,
                                payload=payload,
                                position=pos,
                                random_state=self.seed
                            )
                            
                            self.output_records.append({
                                "id": f"{SOURCE_BIPIA}_{actual_split}_attack_{attack_idx}",
                                "context": poisoned_text,
                                "attack_instruction": payload,
                                "label": Label.ATTACK.value,
                                "attack_family": family,
                                "attack_position": pos,
                                "split": actual_split,
                                "task": self.task,
                                "source": SOURCE_BIPIA,
                            })
                            attack_idx += 1



        df = pd.DataFrame(self.output_records)
        
        # 3. Balancing Check & Training Attack Sampler
        df = self._handle_balancing(df)
        
        return df

    def _handle_balancing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Report on class imbalance and apply training attack sampling if configured."""
        if df.empty or "split" not in df.columns:
            return df
            
        train_df = df[df["split"] == Split.TRAIN.value]
        if len(train_df) == 0:
            return df
            
        n_benign = len(train_df[train_df["label"] == Label.BENIGN.value])
        n_attack = len(train_df[train_df["label"] == Label.ATTACK.value])
        
        logger.info(f"Class distribution - Train Benign: {n_benign}, Train Attack: {n_attack}")
        
        if self.target_attack_samples and self.target_attack_samples < n_attack:
            logger.info(f"Sampling training attacks to target {self.target_attack_samples}...")
            
            # Keep all non-training or benign samples
            mask_keep = (df["split"] != Split.TRAIN.value) | (df["label"] == Label.BENIGN.value)
            df_keep = df[mask_keep]
            
            # Extract training attacks
            train_attacks = df[(df["split"] == Split.TRAIN.value) & (df["label"] == Label.ATTACK.value)]
            
            # Grouped proportional sampling
            groups = train_attacks.groupby(["attack_family", "attack_position"])
            n_groups = groups.ngroups
            n_per_group = max(1, self.target_attack_samples // n_groups)
            
            sampled_attacks = groups.sample(n=n_per_group, random_state=self.seed)
            
            # Exact target adjustment (if rounding left a few off)
            if len(sampled_attacks) < self.target_attack_samples:
                remaining = self.target_attack_samples - len(sampled_attacks)
                unsampled = train_attacks.drop(sampled_attacks.index)
                if not unsampled.empty:
                    extra = unsampled.sample(n=min(remaining, len(unsampled)), random_state=self.seed)
                    sampled_attacks = pd.concat([sampled_attacks, extra])
            elif len(sampled_attacks) > self.target_attack_samples:
                sampled_attacks = sampled_attacks.sample(n=self.target_attack_samples, random_state=self.seed)
                
            df = pd.concat([df_keep, sampled_attacks], ignore_index=True)
            
            new_train_attack = len(sampled_attacks)
            logger.info(f"Training attacks sampled from {n_attack} down to {new_train_attack}.")
        else:
            if n_attack > 0 and (n_benign / n_attack) < 0.1:
                logger.warning(f"Severe class imbalance detected ({n_benign} benign vs {n_attack} attacks).")
                
        return df

    def save_reports(self, output_dir: Union[str, Path], df: pd.DataFrame) -> None:
        """Generate dataset_statistics.json and construction_config.yaml."""
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # 1. Dataset Statistics
        stats = {
            "train_benign_count": len(df[(df["split"] == Split.TRAIN.value) & (df["label"] == Label.BENIGN.value)]),
            "train_attack_count": len(df[(df["split"] == Split.TRAIN.value) & (df["label"] == Label.ATTACK.value)]),
            "validation_benign_count": len(df[(df["split"] == Split.VALIDATION.value) & (df["label"] == Label.BENIGN.value)]),
            "validation_attack_count": len(df[(df["split"] == Split.VALIDATION.value) & (df["label"] == Label.ATTACK.value)]),
            "test_benign_count": len(df[(df["split"] == Split.TEST.value) & (df["label"] == Label.BENIGN.value)]),
            "test_attack_count": len(df[(df["split"] == Split.TEST.value) & (df["label"] == Label.ATTACK.value)]),
            "attack_family_distribution": df[df["label"] == Label.ATTACK.value]["attack_family"].value_counts().to_dict(),
            "attack_position_distribution": df[df["label"] == Label.ATTACK.value]["attack_position"].value_counts().to_dict(),
        }
        
        with open(output_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
            
        # 2. Construction Config
        config = {
            "task": self.task,
            "positions": self.positions,
            "balance": self.balance,
            "seed": self.seed,
            "benchmark_version": "official-bipia",
        }
        
        with open(output_dir / "construction_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)

        logger.info(f"Saved dataset_statistics.json and construction_config.yaml to {output_dir}")
