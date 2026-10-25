import logging


class LevelFormatter(logging.Formatter):
	FORMATS = {  # noqa: RUF012
		logging.DEBUG:    '### DEBUG - %(asctime)s - %(funcName)s - %(message)s',
		logging.INFO:     '### INFO - %(asctime)s - %(funcName)s - %(message)s',
		logging.ERROR:    '### ERROR - %(asctime)s - %(funcName)s - %(message)s',
	}

	def format(self, record):
		fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.DEBUG])
		formatter = logging.Formatter(fmt)
		return formatter.format(record)