"""Explicit deterministic carry command slew; no claim about motor acceleration."""
from copy import deepcopy
import hashlib
import numpy as np

PROTOCOL='learned-carry-command-slew-v1'


def native_control(action):
    """Mirror the unchanged carry adapter's clip and L2 normalization."""
    control=np.asarray(action,dtype=float)
    if control.shape!=(2,) or not np.isfinite(control).all():raise ValueError('Two finite carry controls required')
    control=np.clip(control,-1.,1.)
    control/=max(1.,float(np.linalg.norm(control)))
    return control


def slew_control(action,previous,limit=.18):
    desired=native_control(action);previous=np.asarray(previous,dtype=float)
    if previous.shape!=(2,) or not np.isfinite(previous).all() or np.linalg.norm(previous)>1.+1e-12:
        raise ValueError('Previous effective control must be in the unit disk')
    if not np.isfinite(limit) or limit<=0:raise ValueError('Positive finite command slew limit required')
    change=desired-previous;size=float(np.linalg.norm(change))
    fraction=min(1.,limit/max(size,1e-30))
    output=np.asarray(previous+change*fraction,dtype=np.float32)
    effective=native_control(output)
    # LearnedSystem casts every action to float32. Keep the actual post-cast
    # control as history; one fixed inward numerical correction prevents an
    # outward float32 rounding error from violating the declared bound.
    if np.linalg.norm(effective-previous)>limit:
        output=np.asarray(previous+change*fraction*(1.-2e-6),dtype=np.float32)
        effective=native_control(output)
    if np.linalg.norm(effective-previous)>limit+1e-12:raise AssertionError('Effective slew bound violated')
    return output,effective


class CommandSlewPolicy:
    """One native deterministic prediction; raw baseline preserves its exact object."""
    def __init__(self,model,*,filtered=False,limit=.18,persist=None):
        self.model=model;self.filtered=bool(filtered);self.limit=float(limit);self.persist=persist
        self.previous=np.zeros(2,dtype=float);self.records=[];self.integrity_error=None
        self.action_space=model.action_space;self.observation_space=model.observation_space
    def predict(self,observation,deterministic=True):
        try:
            if deterministic is not True:raise ValueError('This diagnostic uses deterministic deployment only')
            prediction=self.model.predict(observation,deterministic=True)
            raw=np.asarray(prediction[0]);native_input=np.asarray(raw,dtype=np.float32);desired=native_control(native_input);before=self.previous.copy()
            if self.filtered:output,effective=slew_control(native_input,before,self.limit)
            else:output=prediction[0];effective=desired
            h=hashlib.sha256()
            for key,value in sorted(observation.items()):
                array=np.asarray(value);h.update(key.encode());h.update(str((array.dtype.str,array.shape)).encode());h.update(array.tobytes())
            row={'decision':len(self.records),'raw_action':raw.copy(),'native_raw_control':desired.copy(),
                'previous_effective_requested_control':before,'returned_action':np.asarray(output).copy(),'effective_requested_control':effective.copy(),
                'command_change_l2':float(np.linalg.norm(effective-before)),
                'raw_to_requested_l2':float(np.linalg.norm(desired-effective)),
                'observation_sha256':h.hexdigest(),'filtered':self.filtered,'limit':self.limit if self.filtered else None}
            self.records.append(row)
            if self.persist:self.persist(deepcopy(row))
            self.previous=effective.copy()
            return (output,prediction[1]) if self.filtered else prediction
        except BaseException as error:self.integrity_error=repr(error);raise
