#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from loguru import logger
from tqdm import tqdm

#  from rich import print


def do_new(args):
    logger.warning(f"do_new")
    latest_dir = args.latest_dir

    import shutil
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    os.makedirs(latest_dir)

    import uuid
    local_id = str(uuid.uuid1()).replace('-', '')[:8]
    local_id_file = os.path.join(latest_dir, "local_id")
    with open(local_id_file, 'w') as wt:
        wt.write(f"{local_id}")
    logger.warning(f"New model id: {local_id}")

    args.local_id = local_id
    args.local_dir = os.path.join(args.saved_models_path, args.local_id)
    from .utils import save_args
    save_args(args, latest_dir)

    #  cmd_cp_params =
    #      f"cp {theta_src_path}/templates/{app_type}/{app_type}_params.py {latest_dir}/{dataset_name}_params.py"
    #  logger.warning(f"{cmd_cp_params}")

    logger.warning(f"dataset_name: {args.dataset_name}")
    logger.warning(f"experiment_name: {args.experiment_name}")
    logger.warning(f"local_id: {args.local_id}")
    logger.warning(f"local_dir: {args.local_dir}")
    logger.warning(f"latest_dir: {args.latest_dir}")
    logger.warning(f"saved_models_path: {args.saved_models_path}")


class GlueApp:
    def __init__(self, experiment_params, glue_labels, add_special_args=None):
        self.experiment_params = experiment_params
        self.glue_labels = glue_labels

        from .glue import load_model, get_args

        args = get_args(experiment_params=experiment_params,
                        special_args=[add_special_args])
        args.app_type = 'glue'
        self.args = args
        logger.info(f"args: {args}")
        #  print("[bold cyan]args:[/bold cyan]", args)

        self.trainer = None

    def get_trainer(self):
        # -------------------- Model --------------------
        if self.trainer is None:
            args = self.args

            from .glue import GlueTrainer

            class AppTrainer(GlueTrainer):
                def __init__(self, args, glue_labels):
                    super(AppTrainer, self).__init__(args,
                                                     glue_labels,
                                                     build_model=None)

                #  def on_predict_end(self, args, test_dataset):
                #      super(Trainer, self).on_predict_end(args, test_dataset)

            self.trainer = AppTrainer(args, self.glue_labels)

        return self.trainer

    def load_model(self):
        from .glue import load_model as global_load_model

        model = global_load_model(self.args)
        return model

    def run(self,
            train_data_generator,
            test_data_generator,
            generate_submission=None,
            eval_data_generator=None):
        args = self.args

        assert train_data_generator is not None
        assert test_data_generator is not None

        #  if eval_data_generator is None:
        #      eval_data_generator = train_data_generator

        def do_eda(args):
            from .glue_utils import show_glue_datainfo
            show_glue_datainfo(self.glue_labels, train_data_generator,
                               args.train_file, test_data_generator,
                               args.test_file)

        def do_submit(args):
            submission_file = None
            if generate_submission:
                submission_file = generate_submission(args)
            from .utils import archive_local_model
            archive_local_model(args, submission_file)

        def do_generate_submission(args):
            if generate_submission is not None:
                if args.dataset_file is None:
                    logger.error(f"dataset_file parameter must be set.")
                    return
                if args.submission_file is None:
                    logger.error(f"submission_file parameter must be set.")
                    return
                generate_submission(args,
                                    reviews_file=args.dataset_file,
                                    submission_file=args.submission_file)

        if args.do_new:
            do_new(args)

        elif args.do_eda:
            do_eda(args)

        elif args.do_submit:
            do_submit(args)

        elif args.generate_submission:
            do_generate_submission(args)

        else:

            trainer = self.get_trainer()

            from .glue_utils import load_train_val_examples, load_test_examples

            def do_train(args):
                if eval_data_generator is None:
                    train_examples, val_examples = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.glue_labels,
                        shuffle=True,
                        train_rate=args.train_rate,
                        num_augments=args.num_augments)
                else:
                    train_examples, _ = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.glue_labels,
                        shuffle=True,
                        train_rate=1.0,
                        num_augments=args.num_augments)

                    _, val_examples = load_train_val_examples(
                        args,
                        eval_data_generator,
                        self.glue_labels,
                        shuffle=False,
                        train_rate=0.0,
                        num_augments=0)
                trainer.train(args, train_examples, val_examples)

            def do_eval(args):
                args.model_path = args.best_model_path

                if args.eval_data_generator is None:
                    _, eval_examples = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.glue_labels,
                        shuffle=True,
                        train_rate=args.train_rate,
                        num_augments=args.num_augments)
                else:
                    _, eval_examples = load_train_val_examples(
                        args,
                        eval_data_generator,
                        self.glue_labels,
                        shuffle=False,
                        train_rate=0.0,
                        num_augments=0)

                model = self.load_model()
                trainer.evaluate(args, model, eval_examples)

            def do_predict(args):

                args.model_path = args.best_model_path
                test_examples = load_test_examples(args, test_data_generator)
                model = self.load_model()
                trainer.predict(args, model, test_examples)
                from .glue_utils import save_glue_preds
                save_glue_preds(args,
                                trainer.pred_results,
                                test_examples,
                                probs=trainer.pred_probs)

            if args.do_train:
                do_train(args)

            elif args.do_eval:
                do_eval(args)

            elif args.do_predict:
                do_predict(args)

            elif args.do_experiment:
                #  import mlflow
                #  from .utils import log_global_params
                #  if args.tracking_uri:
                #      mlflow.set_tracking_uri(args.tracking_uri)
                #  mlflow.set_experiment(args.experiment_name)
                #
                #  with mlflow.start_run(run_name=f"{args.local_id}") as mlrun:
                #      log_global_params(args, self.experiment_params)
                if True:

                    # ----- Train -----
                    do_train(args)

                    # ----- Predict -----
                    do_predict(args)

                    # ----- Submit -----
                    do_submit(args)


class MultiLabelsApp(GlueApp):
    def get_trainer(self):
        # -------------------- Model --------------------
        if self.trainer is None:
            args = self.args

            from .glue.trainer_multilabels import GlueTrainer

            class AppTrainer(GlueTrainer):
                def __init__(self, args, glue_labels):
                    super(AppTrainer, self).__init__(args,
                                                     glue_labels,
                                                     build_model=None)

                #  def on_predict_end(self, args, test_dataset):
                #      super(Trainer, self).on_predict_end(args, test_dataset)

            self.trainer = AppTrainer(args, self.glue_labels)

        return self.trainer


class NerApp:
    def __init__(self,
                 experiment_params,
                 ner_labels,
                 ner_connections,
                 add_special_args=None):
        self.experiment_params = experiment_params
        self.ner_labels = ner_labels
        self.ner_connections = ner_connections

        if experiment_params.ner_params.ner_type == 'span':
            from .ner_span import load_model, get_args, NerTrainer
        elif experiment_params.ner_params.ner_type == 'pn':
            from .ner_pn import load_model, get_args, NerTrainer
        else:
            from .ner import load_model, get_args, NerTrainer

        args = get_args(experiment_params=experiment_params,
                        special_args=[add_special_args])
        args.app_type = 'ner'
        self.args = args
        logger.info(f"args: {args}")
        #  print("[bold cyan]args:[/bold cyan]", args)

        self.trainer = None

    def get_trainer(self):
        # -------------------- Model --------------------
        if self.trainer is None:
            args = self.args
            if args.ner_type == 'span':
                from .ner_span import NerTrainer
            elif args.ner_type == 'pn':
                from .ner_pn import NerTrainer
            else:
                from .ner import NerTrainer

            class AppTrainer(NerTrainer):
                def __init__(self, args, ner_labels):
                    super(AppTrainer, self).__init__(args,
                                                     ner_labels,
                                                     build_model=None)

                #  def on_predict_end(self, args, test_dataset):
                #      super(Trainer, self).on_predict_end(args, test_dataset)

            self.trainer = AppTrainer(args, self.ner_labels)

        return self.trainer

    def load_model(self):
        if self.experiment_params.ner_params.ner_type == 'span':
            from .ner_span import load_model as global_load_model
        elif self.experiment_params.ner_params.ner_type == 'pn':
            from .ner_pn import load_model as global_load_model
        else:
            from .ner import load_model as global_load_model

        model = global_load_model(self.args)
        return model

    def run(self,
            train_data_generator,
            test_data_generator,
            generate_submission=None,
            eval_data_generator=None):
        args = self.args

        assert train_data_generator is not None
        assert test_data_generator is not None

        #  if eval_data_generator is None:
        #      eval_data_generator = train_data_generator

        def do_eda(args):
            from .ner_utils import show_ner_datainfo
            show_ner_datainfo(self.ner_labels, train_data_generator,
                              args.train_file, test_data_generator,
                              args.test_file)

        def do_submit(args):
            from .utils import archive_local_model
            archive_local_model(args)
            submission_file = None
            if generate_submission:
                submission_file = generate_submission(args)

        def do_generate_submission(args):
            if generate_submission is not None:
                if args.dataset_file is None:
                    logger.error(f"dataset_file parameter must be set.")
                    return
                if args.submission_file is None:
                    logger.error(f"submission_file parameter must be set.")
                    return
                generate_submission(args,
                                    reviews_file=args.dataset_file,
                                    submission_file=args.submission_file)

        if args.do_new:
            do_new(args)

        if args.do_eda:
            do_eda(args)

        elif args.do_submit:
            do_submit(args)

        elif args.generate_submission:
            do_generate_submission(args)

        elif args.to_train_poplar:
            from .ner_utils import to_train_poplar
            to_train_poplar(args,
                            train_data_generator,
                            ner_labels=ner_labels,
                            ner_connections=ner_connections,
                            start_page=args.start_page,
                            max_pages=args.max_pages)

        elif args.to_reviews_poplar:
            from .ner_utils import to_reviews_poplar
            to_reviews_poplar(args,
                              ner_labels=ner_labels,
                              ner_connections=ner_connections,
                              start_page=args.start_page,
                              max_pages=args.max_pages)

        else:

            trainer = self.get_trainer()

            from .ner_utils import load_train_val_examples, load_test_examples

            def do_train(args):
                if eval_data_generator is None:
                    train_examples, val_examples = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.ner_labels,
                        shuffle=True,
                        train_rate=args.train_rate,
                        num_augments=args.num_augments,
                        aug_train_only=args.aug_train_only)
                else:
                    train_examples, _ = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.ner_labels,
                        shuffle=True,
                        train_rate=1.0,
                        num_augments=args.num_augments,
                        aug_train_only=args.aug_train_only)

                    _, val_examples = load_train_val_examples(
                        args,
                        eval_data_generator,
                        self.ner_labels,
                        shuffle=False,
                        train_rate=0.0,
                        num_augments=0,
                        aug_train_only=args.aug_train_only)
                trainer.train(args, train_examples, val_examples)

            def do_eval(args):
                args.model_path = args.best_model_path

                if eval_data_generator is None:
                    _, eval_examples = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.ner_labels,
                        shuffle=True,
                        train_rate=args.train_rate,
                        num_augments=args.num_augments,
                        aug_train_only=args.aug_train_only)
                else:
                    _, eval_examples = load_train_val_examples(
                        args,
                        eval_data_generator,
                        self.ner_labels,
                        shuffle=False,
                        train_rate=0.0,
                        num_augments=0,
                        aug_train_only=args.aug_train_only)

                model = self.load_model()
                trainer.evaluate(args, model, eval_examples)

            def do_predict(args):

                from .ner_utils import save_ner_preds
                args.model_path = args.best_model_path
                test_examples = load_test_examples(args, test_data_generator)
                model = self.load_model()
                trainer.predict(args, model, test_examples)
                reviews_file, category_mentions_file = save_ner_preds(
                    args, trainer.pred_results, test_examples)
                return reviews_file, category_mentions_file

            if args.do_train:
                do_train(args)

            if args.do_eval:
                do_eval(args)

            if args.do_predict:
                do_predict(args)

            if args.do_experiment:
                #  import mlflow
                #  from .utils import log_global_params
                #  if args.tracking_uri:
                #      mlflow.set_tracking_uri(args.tracking_uri)
                #  mlflow.set_experiment(args.experiment_name)
                #
                #  with mlflow.start_run(run_name=f"{args.local_id}") as mlrun:
                #      log_global_params(args, self.experiment_params)
                if True:

                    # ----- Train -----
                    do_train(args)

                    # ----- Predict -----
                    do_predict(args)

                    # ----- Submit -----
                    do_submit(args)


class SpoApp:
    def __init__(self,
                 experiment_params,
                 predicate_labels,
                 add_special_args=None):
        self.experiment_params = experiment_params
        self.predicate_labels = predicate_labels

        from .re_casrel import load_model, get_args, SpoTrainer

        args = get_args(experiment_params=experiment_params,
                        special_args=[add_special_args])
        args.app_type = 'spo'
        self.args = args
        logger.info(f"args: {args}")
        #  print("[bold cyan]args:[/bold cyan]", args)

        self.trainer = None

    def get_trainer(self):
        # -------------------- Model --------------------
        if self.trainer is None:
            args = self.args
            from .re_casrel import SpoTrainer

            class AppTrainer(SpoTrainer):
                def __init__(self, args, predicate_labels):
                    super(AppTrainer, self).__init__(args,
                                                     predicate_labels,
                                                     build_model=None)

                #  def on_predict_end(self, args, test_dataset):
                #      super(Trainer, self).on_predict_end(args, test_dataset)

            self.trainer = AppTrainer(args, self.predicate_labels)

        return self.trainer

    def load_model(self):
        from .re_casrel import load_model as global_load_model

        model = global_load_model(self.args)
        return model

    def run(self,
            train_data_generator,
            test_data_generator,
            generate_submission=None,
            eval_data_generator=None):
        args = self.args

        assert train_data_generator is not None
        assert test_data_generator is not None

        #  if eval_data_generator is None:
        #      eval_data_generator = train_data_generator

        def do_eda(args):
            from .spo_utils import show_spo_datainfo
            show_spo_datainfo(self.predicate_labels, train_data_generator,
                              args.train_file, test_data_generator,
                              args.test_file)

        def do_submit(args):
            submission_file = None
            if generate_submission:
                submission_file = generate_submission(args)
            from .utils import archive_local_model
            archive_local_model(args, submission_file)

        if args.do_new:
            do_new(args)

        elif args.do_eda:
            do_eda(args)

        elif args.do_submit:
            do_submit(args)

        else:

            trainer = self.get_trainer()

            from .spo_utils import load_train_val_examples, load_test_examples

            def do_train(args):
                if eval_data_generator is None:
                    train_examples, val_examples = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.predicate_labels,
                        shuffle=True,
                        train_rate=args.train_rate,
                        num_augments=args.num_augments)
                else:
                    train_examples, _ = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.predicate_labels,
                        shuffle=True,
                        train_rate=1.0,
                        num_augments=args.num_augments)

                    _, val_examples = load_train_val_examples(
                        args,
                        eval_data_generator,
                        self.predicate_labels,
                        shuffle=False,
                        train_rate=0.0,
                        num_augments=0)

                trainer.train(args, train_examples, val_examples)

            def do_eval(args):
                args.model_path = args.best_model_path

                if eval_data_generator is None:
                    _, eval_examples = load_train_val_examples(
                        args,
                        train_data_generator,
                        self.predicate_labels,
                        shuffle=True,
                        train_rate=args.train_rate,
                        num_augments=args.num_augments)
                else:
                    _, eval_examples = load_train_val_examples(
                        args,
                        eval_data_generator,
                        self.predicate_labels,
                        shuffle=False,
                        train_rate=0.0,
                        num_augments=0)

                model = self.load_model()
                trainer.evaluate(args, model, eval_examples)

            def do_predict(args):

                args.model_path = args.best_model_path
                test_examples = load_test_examples(args, test_data_generator)
                model = self.load_model()
                trainer.predict(args, model, test_examples)
                from .spo_utils import save_spo_preds
                save_spo_preds(args, trainer.pred_results, test_examples)

            if args.do_train:
                do_train(args)

            elif args.do_eval:
                do_eval(args)

            elif args.do_predict:
                do_predict(args)

            elif args.do_experiment:
                #  import mlflow
                #  from .utils import log_global_params
                #  if args.tracking_uri:
                #      mlflow.set_tracking_uri(args.tracking_uri)
                #  mlflow.set_experiment(args.experiment_name)
                #
                #  with mlflow.start_run(run_name=f"{args.local_id}") as mlrun:
                #      log_global_params(args, self.experiment_params)
                if True:

                    # ----- Train -----
                    do_train(args)

                    # ----- Predict -----
                    do_predict(args)

                    # ----- Submit -----
                    do_submit(args)
