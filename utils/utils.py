from collections import Counter
import logging

def log_counters(train_counter: Counter, test_counter: Counter) -> None:
    """Log the counts of tasks in train and test data
    """
    logger = logging.getLogger(__name__)
    for k, v in train_counter.items():
        logger.info("[Train] %s\t%d" % (k, v))
    for k, v in test_counter.items():
        logger.info("[Test] %s\t%d" % (k, v))

    logger.info("%d train, %d test" % (len(train_counter), len(test_counter)))

def init_counters(train_data: list, test_data: list) -> tuple:
    """Initiate train and test counters to cound tasks

    Returns:
        tuple<Counter>: train_counter, test_counter
    """
    train_counter = Counter()
    test_counter = Counter()
    for dp in train_data:
        train_counter[dp["task"]] += 1
    for dp in test_data:
        test_counter[dp["task"]] += 1
    return train_counter, test_counter
