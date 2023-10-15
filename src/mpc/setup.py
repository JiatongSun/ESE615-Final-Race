from setuptools import setup

package_name = 'mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kedar Prasad Karpe, Griffon McMahon, Jiatong Sun',
    maintainer_email='karpenet@seas.upenn.edu, gmcmahon@seas.upenn.edu, jtsun@seas.upenn.edu',
    description='f1tenth mpc lab',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_node = mpc.mpc_node:main',
        ],
    },
)
