import logging

logging_set = False
def config(logger):
    global logging_set
    if logging_set:
        return
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    file_handler = logging.FileHandler('logs/output.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging_set = True