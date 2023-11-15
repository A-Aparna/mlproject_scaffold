from setuptools import setup, find_packages
from typing import List

def get_requirements(path:str)->List[str]:
    with open(path, 'r') as f:
        r= f.read().splitlines()

        r.remove('-e .')
        return 
print(get_requirements('requirements.txt'))

setup(
    name='mlpipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)