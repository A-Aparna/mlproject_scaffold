import os
os.chdir('C:\\Aparna\\mlops')
from src.logger import logging

#from helper import hello
from inference import inference

print(os.listdir())
from src.modules.data_transformation import load

logging.info("testing")