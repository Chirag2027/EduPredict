# This setup.py is responsible in creating my ML Application as a package. 

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .' 

def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements] # remove the newline character

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'EduPredict',
    version = '0.0.1',
    author = 'Chirag Verma',
    author_email = 'chirag.yep@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)