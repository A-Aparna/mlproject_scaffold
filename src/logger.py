import os
import logging
from datetime import datetime
from src.helper import hello


timestamp = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
logs_dir = os.path.join(os.getcwd(),'logs')
os.makedirs(logs_dir,exist_ok=True)

logfilename = os.path.join(logs_dir,f'{timestamp}.log')

logging.basicConfig(filename=logfilename,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
