import datetime
import logging
from config.local_settings import project_id
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from config.local_settings import firestore_creds
import json
from google.oauth2 import service_account

creds_filename = '/tmp/creds.json'
with open(creds_filename, 'w') as f:
	json.dump(firestore_creds, f)

def submit_job(
	image_url=None,
	scale_tier = None, 
	extra_args=None, 
	master_type=None, 
	accelerator_type=None,
	workers=None
):
	
	job_name = "evolve_neat_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
	args = []

	if extra_args:
		args += extra_args

	if not scale_tier:
		scale_tier = 'BASIC'

	training_inputs = {
		'args': args,
		'pythonVersion': '3.7',
		'scaleTier': scale_tier,
		'region': 'us-west1',
		'jobDir': "gs://eth-neat/cfz",
	}

	training_inputs.update({'packageUris': ["gs://eth-neat/cfz/evolve_eth_neat-0.1.tar.gz"],
								'pythonModule': 'scripts.evolve',
								'runtimeVersion': '2.1',})

	if scale_tier == 'CUSTOM':
		training_inputs.update({'masterType': master_type})

	if workers:
		training_inputs.update({
			'workerType': master_type,
			'workerCount': workers
		})

	if accelerator_type:
		training_inputs.update({'masterConfig': {'acceleratorConfig': {'count': 1, 'type': accelerator_type}}})
	
	scopes = ['https://www.googleapis.com/auth/cloud-platform']
	creds = service_account.Credentials.from_service_account_file(
		creds_filename, 
		scopes=scopes
	)
	job_spec = {"jobId": job_name, "trainingInput": training_inputs}
	cloudml = discovery.build("ml", "v1", cache_discovery=False, credentials=creds)
	request = cloudml.projects().jobs().create(body=job_spec, parent=f'projects/{project_id}')
	try:
		response = request.execute()
	except HttpError as err:
		logging.error('There was an error creating the training job.'
					  ' Check the details:')
		logging.error(err._get_reason())

def evolve_Task(event="", context="",):
	return submit_job(
		# accelerator_type='NVIDIA-TESLA-K80', 
		master_type='n1-standard-16', 
		scale_tier='CUSTOM',
		extra_args=[
			'--prefix', 'test12_futuresWeekly', 
			'--config', 'data/config-feedforward-1kpop-1.05Goal',
			'--data_files', 'data/ETHUSDT_15m_data_TA_PCA_futures_2021-12-1_2022-9-30.csv,data/BTCUSDT_15m_data_TA_PCA_futures_2021-12-1_2022-9-30.csv',
			'--fitness_metric', 'median_avg_roi',
			'--output_int', '1',
			'--n_inputs', '12',
			'--n_outputs', '17',
			'--interval', 'weekly',
			'--data_interval', '15m',
			'--fitness_threshold', '1.10',
			'--hidden_layers', '2',
			'--pop_size', '500',
			'--max_stag', '7',
			'--capital', '1000000',
			'--n_cv', '30',
			'--btc', 
			'--eth', 
			'--pca'
		]
	)
