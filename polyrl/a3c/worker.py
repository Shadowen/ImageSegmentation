#!/usr/bin/env python
import argparse
import logging
import os
import signal
import sys
import time

import tensorflow as tf

from polyrl.a3c import A3C
from polyrl.polygon_env import PolygonEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(args, server):
    env = PolygonEnv()
    trainer = A3C(env, args.task, args.visualise)

    # Connect to parameter server and to own server
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    logger.info("Events directory: {}_{}".format(logdir, args.task))

    local_variables, global_variables = [], []
    for v in tf.global_variables():
        local_variables.append(v) if v.name.startswith('local') else global_variables.append(v)
    logger.info('Local vars:')
    for v in local_variables:
        logger.info('  %s %s', v.name, v.get_shape())
    logger.info('Global vars:')
    for v in global_variables:
        logger.info('  %s %s', v.name, v.get_shape())
    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=6)
    is_chief = (args.task == 0)
    sv = tf.train.Supervisor(is_chief=is_chief, logdir=logdir, saver=saver, summary_op=None,
                             summary_writer=summary_writer, global_step=trainer.global_step, save_model_secs=1800,
                             save_summaries_secs=120, init_op=tf.variables_initializer(global_variables),
                             ready_op=tf.report_uninitialized_variables(global_variables),
                             local_init_op=tf.variables_initializer(local_variables) if is_chief else trainer.sync)

    num_global_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " "One "
        "common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(master=server.target, config=config) as sess, sess.as_default():
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('Reached {} steps. Worker stopped.'.format(global_step))


def cluster_spec(num_workers, num_ps):
    """
    More tensorflow setup for data parallelism
    """
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """
    Setting up Tensorflow for data parallel work
    """

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128 + signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    tf.app.run()
