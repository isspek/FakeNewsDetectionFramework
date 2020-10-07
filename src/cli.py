from logger import logger
from data_reader import read_constraint_splits
from annotate import annotate


if __name__ == "__main__":
    logger.info('Detection of fake news on social media')

    logger.info('Reading competation dataset')
    constraint_data = read_constraint_splits()
    val = constraint_data['val']
    annotate(val)