import torch
import tensorflow as tf
import os
import logging
from pathlib import Path

import numpy as np

def normalize_np(x):
  x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
  return x_norm

def clear(x, normalize=True):
  x = x.detach().cpu().squeeze().numpy()
  if normalize:
    x = normalize_np(x)
  return x






def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state, name="checkpoint.pth"):
  ckpt_dir = Path(ckpt_dir)
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir / name)