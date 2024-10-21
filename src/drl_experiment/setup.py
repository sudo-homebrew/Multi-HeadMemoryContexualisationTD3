from setuptools import setup

package_name = 'drl_experiment'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunghjopnam',
    maintainer_email='70418121+sudo-homebrew@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drl_experiment1 = drl_experiment.drl_test1:main',
            'drl_experiment2 = drl_experiment.drl_test2:main',
            'record_obs = drl_experiment.record_obstacles:main'
        ],
    },
)
