#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:48:40 2021

@author: krasax
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import re

import tensorflow.compat.v1 as tf

from object_detection import inputs
from object_detection.builders import optimizer_builder
from object_detection.utils import config_util
from object_detection import model_lib_v2

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def evaluate_all_checkpoints(
    pipeline_config_path,
    config_override=None,
    train_steps=None,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=1,
    use_tpu=False,
    override_eval_num_epochs=True,
    postprocess_on_cpu=False,
    model_dir=None,
    checkpoint_dir=None,
    wait_interval=180,
    timeout=100,
    eval_index=0,
    save_final_config=False,
    **kwargs):
  """Run continuous evaluation of a detection model eagerly.

  This method builds the model, and continously restores it from the most
  recent training checkpoint in the checkpoint directory & evaluates it
  on the evaluation data.

  Args:
    pipeline_config_path: A path to a pipeline config file.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    sample_1_of_n_eval_examples: Integer representing how often an eval example
      should be sampled. If 1, will sample all examples.
    sample_1_of_n_eval_on_train_examples: Similar to
      `sample_1_of_n_eval_examples`, except controls the sampling of training
      data for evaluation.
    use_tpu: Boolean, whether training and evaluation should run on TPU.
    override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for
      eval_input.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true,
      postprocess is scheduled on the host cpu.
    model_dir: Directory to output resulting evaluation summaries to.
    checkpoint_dir: Directory that contains the training checkpoints.
    wait_interval: The mimmum number of seconds to wait before checking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait for a checkpoint. Execution
      will terminate if no new checkpoints are found after these many seconds.
    eval_index: int, If given, only evaluate the dataset at the given
      index. By default, evaluates dataset at 0'th index.
    save_final_config: Whether to save the pipeline config file to the model
      directory.
    **kwargs: Additional keyword arguments for configuration override.
  """
  get_configs_from_pipeline_file = model_lib_v2.MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  create_pipeline_proto_from_configs = model_lib_v2.MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']
  merge_external_params_with_configs = model_lib_v2.MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  if train_steps is not None:
    kwargs['train_steps'] = train_steps
  if override_eval_num_epochs:
    kwargs.update({'eval_num_epochs': 1})
    tf.logging.warning(
        'Forced number of epochs for all eval validations to be 1.')
  configs = merge_external_params_with_configs(
      configs, None, kwargs_dict=kwargs)
  if model_dir and save_final_config:
    tf.logging.info('Saving pipeline config file to directory {}'.format(
        model_dir))
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, model_dir)

  model_config = configs['model']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_configs = configs['eval_input_configs']
  eval_on_train_input_config = copy.deepcopy(train_input_config)
  eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)
  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    tf.logging.warning('Expected number of evaluation epochs is 1, but '
                       'instead encountered `eval_on_train_input_config'
                       '.num_epochs` = '
                       '{}. Overwriting `num_epochs` to 1.'.format(
                           eval_on_train_input_config.num_epochs))
    eval_on_train_input_config.num_epochs = 1

  if kwargs['use_bfloat16']:
    tf.compat.v2.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  eval_input_config = eval_input_configs[eval_index]
  strategy = tf.compat.v2.distribute.get_strategy()
  with strategy.scope():
    detection_model = model_lib_v2.MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
        model_config=model_config, is_training=True)

  eval_input = strategy.experimental_distribute_dataset(
      inputs.eval_input(
          eval_config=eval_config,
          eval_input_config=eval_input_config,
          model_config=model_config,
          model=detection_model))

  global_step = tf.compat.v2.Variable(
      0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

  optimizer, _ = optimizer_builder.build(
      configs['train_config'].optimizer, global_step=global_step)
  checkpoint_list=[os.path.join(checkpoint_dir,x[:-6]) for x in os.listdir(checkpoint_dir) if x.endswith('.index')]
  for n, latest_checkpoint in enumerate(natural_sort(checkpoint_list)):
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=detection_model, optimizer=optimizer)
    print(f'Evaluation checkpoint {os.path.split(latest_checkpoint)[1]} (Checkpoint {n} of {len(checkpoint_list)}).')
    # We run the detection_model on dummy inputs in order to ensure that the
    # model and all its variables have been properly constructed. Specifically,
    # this is currently necessary prior to (potentially) creating shadow copies
    # of the model variables for the EMA optimizer.
    if eval_config.use_moving_averages:
      unpad_groundtruth_tensors = (eval_config.batch_size == 1 and not use_tpu)
      model_lib_v2._ensure_model_is_built(detection_model, eval_input,
                             unpad_groundtruth_tensors)
      optimizer.shadow_copy(detection_model)

    ckpt.restore(latest_checkpoint).expect_partial()

    if eval_config.use_moving_averages:
      optimizer.swap_weights()

    summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(model_dir, 'eval', eval_input_config.name))
    with summary_writer.as_default():
      model_lib_v2.eager_eval_loop(
          detection_model,
          configs,
          eval_input,
          use_tpu=use_tpu,
          postprocess_on_cpu=postprocess_on_cpu,
          global_step=global_step,
          )