from setuptools import find_packages
from setuptools import setup

setup(
	name='evolve_eth_neat',
	version='0.1',
	packages=find_packages(),
	install_requires=[
		"matplotlib",
		"google-api-core",
		"google-api-python-client",
		"google-auth",
		"google-auth-httplib2",
		"google-auth-oauthlib",
		"google-cloud-core",
		"google-cloud-storage",
		"google-crc32c",
		"google-resumable-media",
		"googleapis-common-protos",
		"numpy",
		"pandas",
		"Pillow",
		"gspread",
		"gspread_formatting"

	],
	include_package_data=True
)
