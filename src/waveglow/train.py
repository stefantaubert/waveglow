import time
from logging import Logger
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from waveglow.checkpoint import Checkpoint, get_iteration
from waveglow.dataloader import parse_batch, prepare_trainloader, prepare_valloader
from waveglow.hparams import ExperimentHParams, HParams, OptimizerHParams
from waveglow.logger import WaveglowLogger
from waveglow.model import WaveGlow
from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.typing import Entries
from waveglow.utils import (SaveIterationSettings, check_save_it, copy_state_dict,
                            get_continue_batch_iteration, get_continue_epoch,
                            get_formatted_current_total, get_pytorch_filename, init_cuddn,
                            init_cuddn_benchmark, init_torch_seed, log_hparams,
                            overwrite_custom_hparams, skip_batch, try_copy_to, validate_model)


class WaveGlowLoss(torch.nn.Module):
  def __init__(self, sigma=1.0):
    super().__init__()
    self.sigma = sigma

  def forward(self, y_pred, y):
    z, log_s_list, log_det_W_list = y_pred
    log_s_total = 0
    log_det_W_total = 0
    for i, log_s in enumerate(log_s_list):
      if i == 0:
        log_s_total = torch.sum(log_s)
        log_det_W_total = log_det_W_list[i]
      else:
        log_s_total = log_s_total + torch.sum(log_s)
        log_det_W_total += log_det_W_list[i]

    loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
    result = loss / (z.size(0) * z.size(1) * z.size(2))
    return result


def load_model(hparams: HParams, state_dict: Optional[dict], device: torch.device) -> WaveGlow:
  model = WaveGlow(hparams)
  model = try_copy_to(model, device)

  if state_dict is not None:
    model.load_state_dict(state_dict)

  return model


def validate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, iteration, wg_logger: WaveglowLogger, logger: Logger) -> None:
  logger.debug("Validating...")
  avg_val_loss, res = validate_model(model, criterion, val_loader, parse_batch)
  logger.info(f"Validation loss {iteration}: {avg_val_loss:9f}")

  logger.debug("Logging to tensorboard...")
  log_only_last_validation_batch = True
  if log_only_last_validation_batch:
    wg_logger.log_validation(*res[-1], iteration)
  else:
    for entry in tqdm(res):
      wg_logger.log_validation(*entry, iteration)
  logger.debug("Finished.")

  return avg_val_loss


def init_torch(hparams: ExperimentHParams) -> None:
  init_torch_seed(hparams.seed)
  init_cuddn(hparams.cudnn_enabled)
  init_cuddn_benchmark(hparams.cudnn_benchmark)


def warm_start_model(model: nn.Module, warm_model: CheckpointWaveglow) -> None:
  copy_state_dict(
    state_dict=warm_model.state_dict,
    to_model=model,
    ignore=[]
  )


def train(custom_hparams: Optional[Dict[str, str]], logdir: Path, trainset: Entries, valset: Entries, save_checkpoint_dir: Path, checkpoint: Optional[CheckpointWaveglow], logger: Logger, warm_model: Optional[CheckpointWaveglow], device: torch.device) -> None:
  complete_start = time.time()
  wg_logger = WaveglowLogger(logdir)

  if checkpoint is not None:
    hparams = checkpoint.get_hparams(logger)
  else:
    hparams = HParams()
  # is it problematic to change the batch size?
  hparams = overwrite_custom_hparams(hparams, custom_hparams)

  log_hparams(hparams, logger)
  init_torch(hparams)

  model, optimizer = load_model_and_optimizer(
    hparams=hparams,
    checkpoint=checkpoint,
    logger=logger,
    device=device,
  )

  iteration = get_iteration(checkpoint)

  if checkpoint is None and warm_model is not None:
    logger.info("Loading states from pretrained model...")
    warm_start_model(model, warm_model)

  criterion = WaveGlowLoss(
    sigma=hparams.sigma
  )

  train_loader = prepare_trainloader(
    hparams=hparams,
    trainset=trainset,
    device=device,
    logger=logger
  )

  val_loader = prepare_valloader(
    hparams=hparams,
    valset=valset,
    device=device,
    logger=logger
  )

  batch_iterations = len(train_loader)
  if batch_iterations == 0:
    logger.error("Not enough training data.")
    raise Exception()

  # Get shared output_directory ready
  # if rank == 0:
  #   if not output_directory.is_dir():
  #     os.makedirs(output_directory)
  #     os.chmod(output_directory, 0o775)
  #   print("output directory", output_directory)

  model.train()

  train_start = time.perf_counter()
  start = train_start

  save_it_settings = SaveIterationSettings(
    epochs=hparams.epochs,
    batch_iterations=batch_iterations,
    save_first_iteration=True,
    save_last_iteration=True,
    iters_per_checkpoint=hparams.iters_per_checkpoint,
    epochs_per_checkpoint=hparams.epochs_per_checkpoint
  )

  # total_its = hparams.epochs * len(train_loader)
  # epoch_offset = max(0, int(iteration / len(train_loader)))
  # # ================ MAIN TRAINING LOOP! ===================
  # for epoch in range(epoch_offset, hparams.epochs):
  #   debug_logger.info("Epoch: {}".format(epoch))
  #   for i, batch in enumerate(train_loader):
  batch_durations: List[float] = []

  continue_epoch = get_continue_epoch(iteration, batch_iterations)
  for epoch in range(continue_epoch, hparams.epochs):
    next_batch_iteration = get_continue_batch_iteration(iteration, batch_iterations)
    skip_bar = None
    if next_batch_iteration > 0:
      logger.debug(f"Current batch is {next_batch_iteration} of {batch_iterations}")
      logger.debug("Skipping batches...")
      skip_bar = tqdm(total=next_batch_iteration)
    for batch_iteration, batch in enumerate(train_loader):
      need_to_skip_batch = skip_batch(
        batch_iteration=batch_iteration,
        continue_batch_iteration=next_batch_iteration
      )
      if need_to_skip_batch:
        assert skip_bar is not None
        skip_bar.update(1)
        #debug_logger.debug(f"Skipped batch {batch_iteration + 1}/{next_batch_iteration + 1}.")
        continue
      # debug_logger.debug(f"Current batch: {batch[0][0]}")

      model.zero_grad()
      x, y = parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      reduced_loss = loss.item()

      loss.backward()

      optimizer.step()

      iteration += 1

      end = time.perf_counter()
      duration = end - start
      start = end

      batch_durations.append(duration)
      logger.info(" | ".join([
        f"Epoch: {get_formatted_current_total(epoch + 1, hparams.epochs)}",
        f"Iteration: {get_formatted_current_total(batch_iteration + 1, batch_iterations)}",
        f"Total iteration: {get_formatted_current_total(iteration, hparams.epochs * batch_iterations)}",
        f"Train loss: {reduced_loss:.6f}",
        f"Duration: {duration:.2f}s/it",
        f"Avg. duration: {np.mean(batch_durations):.2f}s/it",
        f"Total Duration: {(time.perf_counter() - train_start) / 60 / 60:.2f}h"
      ]))

      wg_logger.log_training(reduced_loss, hparams.learning_rate, duration, iteration)

      #wg_logger.add_scalar('training_loss', reduced_loss, iteration)

      save_it = check_save_it(epoch, iteration, save_it_settings)
      if save_it:
        checkpoint = CheckpointWaveglow.from_instances(
          model=model,
          optimizer=optimizer,
          hparams=hparams,
          iteration=iteration,
        )

        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_checkpoint_dir / get_pytorch_filename(iteration)
        checkpoint.save(checkpoint_path, logger)

        validate(model, criterion, val_loader, iteration, wg_logger, logger)

  duration_s = time.time() - complete_start
  logger.info(f'Finished training. Total duration: {duration_s / 60:.2f}m')


def load_optimizer(model_parameters: Iterator[Parameter], hparams: OptimizerHParams, state_dict: Optional[dict]) -> torch.optim.Adam:
  optimizer = torch.optim.Adam(
    params=model_parameters,
    lr=hparams.learning_rate,
  )

  if state_dict is not None:
    optimizer.load_state_dict(state_dict)

  return optimizer


def load_model_and_optimizer(hparams: HParams, checkpoint: Optional[Checkpoint], device: torch.device, logger: Logger) -> None:
  model = load_model(
    hparams=hparams,
    state_dict=checkpoint.state_dict if checkpoint is not None else None,
    device=device,
  )

  optimizer = load_optimizer(
    model_parameters=model.parameters(),
    hparams=hparams,
    state_dict=checkpoint.optimizer if checkpoint is not None else None
  )

  return model, optimizer
