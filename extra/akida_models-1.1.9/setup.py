from os import path
from setuptools import setup, find_packages

# Read the contents of the README file
directory = path.abspath(path.dirname(__file__))
with open(path.join(directory, 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='akida_models',
      version='1.1.9',
      description='Akida Models',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='David Corvoysier',
      author_email='dcorvoysier@brainchip.com',
      url='https://doc.brainchipinc.com',
      license='Apache 2.0',
      license_files=['LICENSE', 'LICENSE.3rdparty'],
      packages=find_packages(),
      entry_points={
        'console_scripts': [
            'akida_models = akida_models.cli:main',
            'utk_face_train = akida_models.utk_face.utk_face_train:main',
            'kws_train = akida_models.kws.kws_train:main',
            'modelnet40_train = akida_models.modelnet40.modelnet40_train:main',
            'yolo_train = akida_models.detection.yolo_train:main',
            'dvs_train = akida_models.dvs.dvs_train:main',
            'cwru_train = akida_models.cwru.cwru_train:main',
            'mnist_train = akida_models.mnist.mnist_train:main',
            'imagenet_train = akida_models.imagenet.imagenet_train:main',
            'portrait128_train = akida_models.portrait128.portrait128_train:main',
            'centernet_train = akida_models.centernet.centernet_train:main'
        ]
      },
      install_requires=['cnn2snn~=2.2.2', 'quantizeml~=0.3.1', 'scipy', 'opencv-python', 'mtcnn',
        'imgaug', 'trimesh', 'tensorflow-addons>=0.18.0'],
      python_requires='>=3.7')
