"""
src/adaptive/teacher_diagnosis.py

Executes a separate diagnostic stage where the Teacher LLM analyzes
the deterministic failure profile and representative failures, acting strictly
as a diagnostician rather than a generator.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


# TODO: Researcher must implement the exact prompt formulation.
DIAGNOSTIC_SYSTEM_PROMPT = """\
You are an expert cybersecurity diagnostic analyst.
Your task is to analyze the failure profile and representative failures of an
instruction-tuned language model designed to detect embedded instruction insertion 
attacks in emails.

You must output your diagnosis in the following exact JSON schema:
{
    "linguistic_patterns": ["pattern 1", "pattern 2", ...],
    "likely_failure_causes": ["cause 1", "cause 2", ...],
    "recommended_curriculum_focus": ["focus 1", "focus 2", ...]
}
"""

# TODO: Researcher must implement the exact prompt formulation.
DIAGNOSTIC_USER_PROMPT_TEMPLATE = """\
FAILURE PROFILE:
{profile_json}

REPRESENTATIVE FAILURES:
{failures_json}

Analyze these failures and provide your diagnosis.
"""


class TeacherDiagnostician:
    """Diagnoses model weaknesses based on failure statistics and samples."""

    def __init__(self, output_dir: Union[str, Path], teacher_client: Any = None):
        """
        Args:
            output_dir: Where to save diagnosis.json.
            teacher_client: The LLM client (e.g., vLLM or similar) used for diagnosis.
        """
        self.output_dir = Path(output_dir)
        self.teacher_client = teacher_client
        ensure_dir(self.output_dir)

    def generate_diagnosis(self, failure_profile_path: Union[str, Path], representative_failures_path: Union[str, Path]) -> None:
        """Read profiling artifacts, query the teacher, and save the diagnosis."""
        with open(failure_profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        failures = []
        with open(representative_failures_path, "r", encoding="utf-8") as f:
            for line in f:
                failures.append(json.loads(line))

        # Format prompt
        # TODO: Implement full prompt templating logic
        user_prompt = DIAGNOSTIC_USER_PROMPT_TEMPLATE.format(
            profile_json=json.dumps(profile, indent=2),
            failures_json=json.dumps(failures, indent=2)
        )

        logger.info("Querying teacher for failure diagnosis...")
        
        messages = [
            {"role": "system", "content": DIAGNOSTIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response_text = self.teacher_client.generate(messages)
        
        # Parse JSON
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        try:
            diagnosis = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse teacher diagnosis JSON: {response_text}")
            diagnosis = {
                "linguistic_patterns": ["Failed to parse patterns"],
                "likely_failure_causes": ["Failed to parse causes"],
                "recommended_curriculum_focus": ["Failed to parse focus"]
            }

        # Save diagnosis
        out_path = self.output_dir / "diagnosis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(diagnosis, f, indent=2)

        logger.info(f"Diagnosis saved to {out_path}")
