import dataclasses
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from logging import Logger, getLogger
from math import floor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.spatial.distance import cosine
from torch import Tensor, nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader
from tqdm import tqdm

_T = TypeVar('_T')
PYTORCH_EXT = ".pt"


def set_torch_thread_to_max() -> None:
  torch.set_num_threads(cpu_count())
  torch.set_num_interop_threads(cpu_count())


def split_hparams_string(hparams: Optional[str]) -> Optional[Dict[str, str]]:
  if hparams is None:
    return None

  assignments = hparams.split(",")
  result = dict([x.split("=") for x in assignments])
  return result


def parse_json(path: Path, encoding: str = 'utf-8') -> Dict:
  assert path.is_file()
  with path.open(mode='r', encoding=encoding) as f:
    tmp = json.load(f)
  return tmp


def overwrite_custom_hparams(hparams_dc: _T, custom_hparams: Optional[Dict[str, str]]) -> _T:
  if custom_hparams is None:
    return hparams_dc

  # custom_hparams = get_only_known_params(custom_hparams, hparams_dc)
  if check_has_unknown_params(custom_hparams, hparams_dc):
    raise Exception()

  set_types_according_to_dataclass(custom_hparams, hparams_dc)

  result = dataclasses.replace(hparams_dc, **custom_hparams)
  return result


def set_types_according_to_dataclass(params: Dict[str, str], hparams: _T) -> None:
  available_params = dataclasses.asdict(hparams)
  for custom_hparam, new_value in params.items():
    assert custom_hparam in available_params.keys()
    hparam_value = available_params[custom_hparam]
    params[custom_hparam] = get_value_in_type(hparam_value, new_value)


def get_only_known_params(params: Dict[str, str], hparams: _T) -> Dict[str, str]:
  available_params = dataclasses.asdict(hparams)
  res = {k: v for k, v in params.items() if k in available_params.keys()}
  return res


def get_value_in_type(old_value: _T, new_value: str) -> _T:
  old_type = type(old_value)
  if new_value == "":
    new_value_with_original_type = None
  else:
    new_value_with_original_type = old_type(new_value)
  return new_value_with_original_type


def check_has_unknown_params(params: Dict[str, str], hparams: _T) -> bool:
  available_params = dataclasses.asdict(hparams)
  for custom_hparam in params.keys():
    if custom_hparam not in available_params.keys():
      return True
  return False


def get_dataclass_from_dict(params: Dict[str, str], dc: Type[_T]) -> Tuple[_T, Set[str]]:
  field_names = {x.name for x in dataclasses.fields(dc)}
  res = {k: v for k, v in params.items() if k in field_names}
  ignored = {k for k in params.keys() if k not in field_names}
  return dc(**res), ignored


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


formatter = logging.Formatter(
  '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
  datefmt='%Y/%m/%d %H:%M:%S'
)


def get_default_logger():
  return logging.getLogger("default")


def prepare_logger(log_file_path: Optional[Path] = None, reset: bool = False, logger: Logger = get_default_logger()) -> None:
  init_logger(logger)
  add_console_out_to_logger(logger)
  if log_file_path is not None:
    if reset:
      reset_file_log(log_file_path)
    add_file_out_to_logger(logger, log_file_path)
  return logger


def init_logger(logger: logging.Logger = get_default_logger()) -> None:
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)
  # disable is required (don't know why) because otherwise DEBUG messages would be ignored!
  logger.manager.disable = logging.NOTSET

  # to disable double logging
  logger.propagate = False

  # take it from the above logger (root)
  logger.setLevel(logging.DEBUG)

  for h in logger.handlers:
    logger.removeHandler(h)

  disable_matplot_logger()
  disable_numba_logger()

  return logger


def disable_matplot_logger():
  disable_matplot_font_logger()
  disable_matplot_colorbar_logger()


def disable_numba_logger():
  disable_numba_core_logger()


def disable_numba_core_logger():
  """
  Disables:
    DEBUG:numba.core.ssa:on stmt: $92load_attr.32 = getattr(value=y, attr=shape)
    DEBUG:numba.core.ssa:on stmt: $const94.33 = const(int, 1)
    DEBUG:numba.core.ssa:on stmt: $96binary_subscr.34 = static_getitem(value=$92load_attr.32, index=1, index_var=$const94.33, fn=<built-in function getitem>)
    DEBUG:numba.core.ssa:on stmt: n_channels = $96binary_subscr.34
    DEBUG:numba.core.ssa:on stmt: $100load_global.35 = global(range: <class 'range'>)
    DEBUG:numba.core.ssa:on stmt: $104call_function.37 = call $100load_global.35(n_out, func=$100load_global.35, args=[Var(n_out, interpn.py:24)], kws=(), vararg=None)
    DEBUG:numba.core.ssa:on stmt: $106get_iter.38 = getiter(value=$104call_function.37)
    DEBUG:numba.core.ssa:on stmt: $phi108.0 = $106get_iter.38
    DEBUG:numba.core.ssa:on stmt: jump 108
    DEBUG:numba.core.byteflow:block_infos State(pc_initial=446 nstack_initial=1):
    AdaptBlockInfo(insts=((446, {'res': '$time_register446.1'}), (448, {'res': '$time_increment448.2'}), (450, {'lhs': '$time_register446.1', 'rhs': '$time_increment448.2', 'res': '$450inplace_add.3'}), (452, {'value': '$450inplace_add.3'}),
    (454, {})), outgoing_phis={}, blockstack=(), active_try_block=None, outgoing_edgepushed={108: ('$phi446.0',)})
    DEBUG:numba.core.byteflow:block_infos State(pc_initial=456 nstack_initial=0):
    AdaptBlockInfo(insts=((456, {'res': '$const456.0'}), (458, {'retval': '$const456.0', 'castval': '$458return_value.1'})), outgoing_phis={}, blockstack=(), active_try_block=None, outgoing_edgepushed={})
    DEBUG:numba.core.interpreter:label 0:
        x = arg(0, name=x)                       ['x']
        y = arg(1, name=y)                       ['y']
        sample_ratio = arg(2, name=sample_ratio) ['sample_ratio']
    ...
  """
  logging.getLogger('numba.core').disabled = True


def disable_matplot_font_logger():
  '''
  Disables:
    DEBUG:matplotlib.font_manager:findfont: score(<Font 'Noto Sans Oriya UI' (NotoSansOriyaUI-Bold.ttf) normal normal 700 normal>) = 10.335
    DEBUG:matplotlib.font_manager:findfont: score(<Font 'Noto Serif Khmer' (NotoSerifKhmer-Regular.ttf) normal normal 400 normal>) = 10.05
    DEBUG:matplotlib.font_manager:findfont: score(<Font 'Samyak Gujarati' (Samyak-Gujarati.ttf) normal normal 500 normal>) = 10.14
    ...
  '''
  logging.getLogger('matplotlib.font_manager').disabled = True


def disable_matplot_colorbar_logger():
  '''
  Disables:
    DEBUG:matplotlib.colorbar:locator: <matplotlib.colorbar._ColorbarAutoLocator object at 0x7f78f08e6370>
    DEBUG:matplotlib.colorbar:Using auto colorbar locator <matplotlib.colorbar._ColorbarAutoLocator object at 0x7f78f08e6370> on colorbar
    DEBUG:matplotlib.colorbar:Setting pcolormesh
  '''
  logging.getLogger('matplotlib.colorbar').disabled = True


def disable_imageio_logger():
  """
  Disables:
    WARNING:imageio:Lossy conversion from float64 to uint8. Range [-0.952922124289318, 1.0000000000043152]. Convert image to uint8 prior to saving to suppress this warning.
    ...
  """
  logging.getLogger('imageio').disabled = True


def init_global_seeds(seed: int) -> None:
  # torch.backends.cudnn.deterministic = True
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
  # only on multi GPU
  # torch.cuda.manual_seed_all(seed)


def add_console_out_to_logger(logger: logging.Logger = get_default_logger()) -> None:
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.NOTSET)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  logger.debug("init console logger")


def add_file_out_to_logger(logger: logging.Logger = get_default_logger(), log_file_path: Path = Path("/tmp/log.txt")) -> None:
  fh = logging.FileHandler(log_file_path)
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.debug(f"init logger to {log_file_path}")


def reset_file_log(log_file_path: Path) -> None:
  if log_file_path.is_file():
    os.remove(log_file_path)


def get_last_checkpoint(checkpoint_dir: Path) -> Tuple[str, int]:
  '''
  Returns the full path of the last checkpoint and its iteration.
  '''
  # checkpoint_dir = get_checkpoint_dir(training_dir_path)
  its = get_all_checkpoint_iterations(checkpoint_dir)
  at_least_one_checkpoint_exists = len(its) > 0
  if not at_least_one_checkpoint_exists:
    raise Exception("No checkpoint iteration found!")
  last_iteration = max(its)
  last_checkpoint = get_pytorch_filename(last_iteration)
  checkpoint_path = checkpoint_dir / last_checkpoint
  return checkpoint_path, last_iteration


def get_all_checkpoint_iterations(checkpoint_dir: Path) -> List[int]:
  filenames = get_filenames(checkpoint_dir)
  checkpoints_str = [get_pytorch_basename(str(x))
                     for x in filenames if is_pytorch_file(str(x))]
  checkpoints = list(sorted(list(map(int, checkpoints_str))))
  return checkpoints


def get_filenames(parent_dir: Path) -> List[Path]:
  assert parent_dir.is_dir()
  _, _, filenames = next(os.walk(parent_dir))
  filenames.sort()
  filenames = [Path(filename) for filename in filenames]
  return filenames


def get_checkpoint(checkpoint_dir: Path, iteration: int) -> str:
  checkpoint_path = checkpoint_dir / get_pytorch_filename(iteration)
  if not checkpoint_path.is_file():
    raise Exception(f"Checkpoint with iteration {iteration} not found!")
  return checkpoint_path


def get_custom_or_last_checkpoint(checkpoint_dir: Path, custom_iteration: Optional[int]) -> Tuple[str, int]:
  return (get_checkpoint(checkpoint_dir, custom_iteration), custom_iteration) if custom_iteration is not None else get_last_checkpoint(checkpoint_dir)


def update_learning_rate_optimizer(optimizer: Optimizer, learning_rate: float) -> None:
  for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate


def copy_state_dict(state_dict: Dict[str, Tensor], to_model: nn.Module, ignore: List[str]) -> None:
  # TODO: ignore as set
  model_dict = {k: v for k, v in state_dict.items() if k not in ignore}
  update_state_dict(to_model, model_dict)


def try_copy_to(x: Union[Tensor, Module], device: torch.device) -> Union[Tensor, Module]:
  try:
    x = x.to(device, non_blocking=True)
  except Exception as ex:
    logger = getLogger(__name__)
    logger.debug(ex)
    logger.warning(f"Mapping to device '{device}' was not successful, therefore using CPU!")
    x = x.to("cpu", non_blocking=True)
  return x


def update_state_dict(model: nn.Module, updates: Dict[str, Tensor]) -> None:
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_hparams(hparams: _T, logger: Logger) -> None:
  logger.info("=== HParams ===")
  for param, val in asdict(hparams).items():
    logger.info(f"- {param} = {val}")
  logger.info("===============")


def get_formatted_current_total(current: int, total: int) -> str:
  return f"{str(current).zfill(len(str(total)))}/{total}"


def validate_model(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, batch_parse_method) -> Tuple[float, Tuple[float, nn.Module, Tuple, Tuple]]:
  model.eval()
  res = []
  with torch.no_grad():
    total_val_loss = 0.0
    # val_loader count is: ceil(validation set length / batch size)
    for batch in tqdm(val_loader):
      x, y = batch_parse_method(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      res.append((reduced_val_loss, model, y, y_pred))
      total_val_loss += reduced_val_loss
    avg_val_loss = total_val_loss / len(val_loader)
  model.train()

  return avg_val_loss, res


@dataclass
class SaveIterationSettings():
  epochs: int
  batch_iterations: int
  save_first_iteration: bool
  save_last_iteration: bool
  iters_per_checkpoint: int
  epochs_per_checkpoint: int


def check_save_it(epoch: int, iteration: int, settings: SaveIterationSettings) -> bool:
  if check_is_first(iteration) and settings.save_first_iteration:
    return True

  if check_is_last(iteration, settings.epochs, settings.batch_iterations) and settings.save_last_iteration:
    return True

  if check_is_save_iteration(iteration, settings.iters_per_checkpoint):
    return True

  is_last_batch_iteration = check_is_last_batch_iteration(iteration, settings.batch_iterations)
  if is_last_batch_iteration and check_is_save_epoch(epoch, settings.epochs_per_checkpoint):
    return True

  return False


def get_next_save_it(iteration: int, settings: SaveIterationSettings) -> Optional[int]:
  result = iteration
  while result <= settings.epochs * settings.batch_iterations:
    epoch = iteration_to_epoch(result, settings.batch_iterations)
    if check_save_it(epoch, result, settings):
      return result
    result += 1
  return None


def check_is_first(iteration: int) -> bool:
  assert iteration >= 0
  # iteration=0 means no training was done yet
  return iteration == 1


def check_is_last(iteration: int, epochs: int, batch_iterations: int) -> bool:
  assert iteration >= 0
  return iteration == epochs * batch_iterations


def check_is_save_iteration(iteration: int, iters_per_checkpoint: int) -> bool:
  assert iteration >= 0
  save_iterations = iters_per_checkpoint > 0
  return iteration > 0 and save_iterations and iteration % iters_per_checkpoint == 0


def check_is_save_epoch(epoch: int, epochs_per_checkpoint: int) -> bool:
  assert epoch >= 0

  save_epochs = epochs_per_checkpoint > 0
  return save_epochs and ((epoch + 1) % epochs_per_checkpoint == 0)


def check_is_last_batch_iteration(iteration: int, batch_iterations: int) -> None:
  assert iteration >= 0
  assert batch_iterations > 0
  if iteration == 0:
    return False
  batch_iteration = iteration_to_batch_iteration(iteration, batch_iterations)
  is_last_batch_iteration = batch_iteration + 1 == batch_iterations
  return is_last_batch_iteration


def get_continue_epoch(current_iteration: int, batch_iterations: int) -> int:
  return iteration_to_epoch(current_iteration + 1, batch_iterations)


def skip_batch(continue_batch_iteration: int, batch_iteration: int) -> None:
  result = batch_iteration < continue_batch_iteration
  return result


def iteration_to_epoch(iteration: int, batch_iterations: int) -> int:
  """result: [0, inf)"""
  # Iteration 0 has no epoch.
  assert iteration > 0

  iteration_zero_based = iteration - 1
  epoch = floor(iteration_zero_based / batch_iterations)
  return epoch


def iteration_to_batch_iteration(iteration: int, batch_iterations: int) -> int:
  """result: [0, iterations)"""
  # Iteration 0 has no batch iteration.
  assert iteration > 0

  iteration_zero_based = iteration - 1
  batch_iteration = iteration_zero_based % batch_iterations
  return batch_iteration


def get_continue_batch_iteration(iteration: int, batch_iterations: int) -> int:
  return iteration_to_batch_iteration(iteration + 1, batch_iterations)


def filter_checkpoints(iterations: List[int], select: Optional[int], min_it: Optional[int], max_it: Optional[int]) -> List[int]:
  if select is None:
    select = 0
  if min_it is None:
    min_it = 0
  if max_it is None:
    max_it = max(iterations)
  process_checkpoints = [checkpoint for checkpoint in iterations if checkpoint %
                         select == 0 and min_it <= checkpoint <= max_it]

  return process_checkpoints


def init_torch_seed(seed: int) -> None:
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def init_cuddn(enabled: bool) -> None:
  torch.backends.cudnn.enabled = enabled


def init_cuddn_benchmark(enabled: bool) -> None:
  torch.backends.cudnn.benchmark = enabled


def get_pytorch_filename(name: Union[str, int]) -> str:
  return f"{name}{PYTORCH_EXT}"


def get_pytorch_basename(filename: str) -> None:
  return filename[:-len(PYTORCH_EXT)]


def is_pytorch_file(filename: str) -> None:
  return filename.endswith(PYTORCH_EXT)


def figure_to_numpy_rgb(figure: Figure) -> np.ndarray:
  figure.canvas.draw()
  data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
  return data


def cosine_dist_mels(a: np.ndarray, b: np.ndarray) -> float:
  a, b = make_same_dim(a, b)
  scores = []
  for channel_nr in range(a.shape[0]):
    channel_a = a[channel_nr]
    channel_b = b[channel_nr]
    score = cosine(channel_a, channel_b)
    if np.isnan(score):
      score = 1
    scores.append(score)
  score = np.mean(scores)
  # scores = cdist(pred_np, orig_np, 'cosine')
  final_score = 1 - score
  return final_score


def make_same_dim(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  dim_a = a.shape[1]
  dim_b = b.shape[1]
  diff = abs(dim_a - dim_b)
  if diff > 0:
    adding_array = np.zeros(shape=(a.shape[0], diff))
    if dim_a < dim_b:
      a = np.concatenate((a, adding_array), axis=1)
    else:
      b = np.concatenate((b, adding_array), axis=1)
  assert a.shape == b.shape
  return a, b
