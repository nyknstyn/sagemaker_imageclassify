from setuptools import setup, find_packages

setup(name='sagemaker-image-classify',
      version='1.0',
      description='Image Classify Using Sagemaker.',
      author='nyknstyn',
      author_email='nicholas.gabriel048@gmail.com',
      url='https://github.com/nyknstyn/',
      packages=find_packages(exclude=('tests', 'docs')))