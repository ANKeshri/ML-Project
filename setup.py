from setuptools import  find_packages, setup
from typing import List


HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[] #list to store the requirements
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()  #read the file
        cleaned_requirements = []
        for req in requirements:
            req = req.split('#')[0].strip()  # Remove comments and whitespace
            if req:  # Only add non-empty lines
                cleaned_requirements.append(req)
        requirements = cleaned_requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT) #remove the -e . from the requirements
    return requirements



#setup function is used to create a package

setup( 
    name='mlproject',
    version='0.0.1',  #version of the package
    author='ANKeshri', 
    author_email='ankeshri2003@gmail.com', 
    packages=find_packages(),  #finds all the packages in the current directory
    install_requires=get_requirements('requirements.txt')        #list of dependencies we cant write 100  of dependencies in the setup.py file so we use a function to get the dependencies from the requirements.txt file
) 