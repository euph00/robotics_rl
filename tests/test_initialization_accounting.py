"""Training-data validation and demonstrations remain distinct counted costs."""
import pytest
from scripts.finetune_initialized import InitializedProgressCallback


def test_preflight_is_counted_once_outside_imported_demonstrations():
    callback=InitializedProgressCallback(None,None)
    metadata={'demonstration_source_action_calls':100,'demonstration_physics_ticks':500,
        'setup_source_action_calls':20,'setup_physics_ticks':100,'initialization_physics_ticks':480,
        'preflight_source_action_calls':30,'preflight_physics_ticks':150}
    record={'incremental_accounting':{'new_evaluation_source_action_calls':10,
        'new_evaluation_physics_ticks':50,'new_setup_source_action_calls':2,
        'new_setup_physics_ticks':10,'new_initialization_physics_ticks':480},
        'new_demonstration_report':{'dataset_metadata':metadata,'transitions_imported':20},
        'legacy_missing':[]}
    callback.add_initialization(record)
    result=callback.accounting()
    assert result['evaluation_source_action_calls']==40
    assert result['demonstration_source_action_calls']==100
    assert result['total_source_action_calls_known']==162
    assert result['total_physics_ticks_known']==1770
    with pytest.raises(ValueError,match='twice'):
        callback.add_initialization(record)
    assert callback.accounting()['total_source_action_calls_known']==162


def test_preflight_is_not_recounted_when_dataset_is_already_inherited():
    callback=InitializedProgressCallback(None,None)
    callback.evaluation_actions=30
    callback.evaluation_ticks=150
    record={'incremental_accounting':{'new_evaluation_source_action_calls':10,
        'new_evaluation_physics_ticks':50,'new_setup_source_action_calls':0,
        'new_setup_physics_ticks':0,'new_initialization_physics_ticks':0},
        'new_demonstration_report':None,'legacy_missing':[]}
    callback.add_initialization(record)
    assert callback.accounting()['evaluation_source_action_calls']==40
