import logging
import yaml

def setup_logger(config):
    level = config["logging"]["level"]
    logfile = config["logging"]["file"]

    logging.basicConfig(
        filename=logfile,
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w"
    )

    return logging.getLogger("intent_eval_logger")
