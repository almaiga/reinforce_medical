"""
Medical Team Components for Medical Self-Play Training
"""

from enum import Enum


class MedicalGameOutcome(Enum):
    """Medical game outcome types for reward calculation"""
    # Attacker outcomes
    ATTACKER_ERROR_UNDETECTED = "attacker_error_undetected"
    ATTACKER_ERROR_DETECTED = "attacker_error_detected"
    ATTACKER_REALISTIC_ERROR = "attacker_realistic_error"
    ATTACKER_UNREALISTIC_ERROR = "attacker_unrealistic_error"

    # Assessor outcomes
    ASSESSOR_CORRECT_DETECTION = "assessor_correct_detection"
    ASSESSOR_MISSED_ERROR = "assessor_missed_error"
    ASSESSOR_FALSE_POSITIVE = "assessor_false_positive"
    ASSESSOR_CORRECT_REJECTION = "assessor_correct_rejection"

    # Other outcomes
    TIE = "tie"
    PARSING_ERROR = "parsing_error"


from .medical_judge import load_medical_judge_model, evaluate_medical_response
from .medical_game_manager import MedicalDialogueGameManager, MedicalGameState
from .remote_judge import (
    create_medical_judge_remote_function,
    test_judge_connection,
    wait_for_judge_server
)

# Try to import OpenRLHF version if available
try:
    from .language_game import MedicalDialogueGameManager as MedicalDialogueGameManagerRL
    _has_openrlhf = True
except ImportError:
    MedicalDialogueGameManagerRL = None
    _has_openrlhf = False

__all__ = [
    'MedicalGameOutcome',
    'load_medical_judge_model',
    'evaluate_medical_response',
    'MedicalDialogueGameManager',
    'MedicalGameState',
    'create_medical_judge_remote_function',
    'test_judge_connection',
    'wait_for_judge_server'
]

if _has_openrlhf:
    __all__.append('MedicalDialogueGameManagerRL')
