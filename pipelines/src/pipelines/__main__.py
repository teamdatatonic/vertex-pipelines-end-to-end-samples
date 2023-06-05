import logging
import traceback
from argparse import ArgumentParser, Namespace
from os import environ as env

import pipelines.configs
from pipelines.utils import compile_pipeline, load_pipeline

ENV_VAR_PIPELINE = "pipeline"
ENV_VAR_CONFIG = "config"


def compile(args: Namespace) -> None:
    for x in ["pipeline", "config"]:
        if getattr(args, x, None) is None:
            raise ValueError(f"argument '{x}' is empty")

    pipelines.configs.config = args.config
    compile_pipeline(load_pipeline(args.pipeline), args.pipeline + ".json")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verbose", "-vv", action="store_true")
    parser.add_argument("--pipeline", default=env.get(ENV_VAR_PIPELINE))
    parser.add_argument("--config", default=env.get(ENV_VAR_CONFIG))
    return parser.parse_args()


def main(args: Namespace) -> None:
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level, format="%(asctime)s %(levelname)-7s: %(message)s"
    )

    try:
        if args.compile:
            compile(args)
        elif args.run:
            raise NotImplementedError("run action is not implemented")
        else:
            raise RuntimeError("no action provided")
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            logging.error(traceback.format_exc())
        else:
            logging.warning("enable verbose logging to print error details")


if __name__ == "__main__":
    main(parse_args())
