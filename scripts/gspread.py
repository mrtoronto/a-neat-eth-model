import json
import logging
import gspread as gs
import gspread_formatting as gsf

from config.local_settings import firestore_creds

def get_worksheet(gc, create_sheet, file_key, sheet_name, overwrite_existing_sheet):
	if create_sheet:
		try:
			ws = gc.open_by_key(file_key).add_worksheet(title=sheet_name, rows="100", cols="20")
		except gs.exceptions.APIError as e:
			ws = gc.open_by_key(file_key).worksheet(sheet_name)
			if overwrite_existing_sheet:
				logging.error(f'Worksheet {sheet_name} already exists. Clearing.')
				ws.clear()
				rules = gsf.get_conditional_format_rules(ws)
				rules.clear()
				rules.save()
			else:
				logging.error(f'Worksheet {sheet_name} already exists. Appending.')

	else:
		try:
			ws = gc.open_by_key(file_key).worksheet(sheet_name)
		except gs.exceptions.WorksheetNotFound as e:
			logging.error(f'Worksheet {sheet_name} not found in {file_key}. Creating...')
			ws = gc.open_by_key(file_key).add_worksheet(title=sheet_name, rows="100", cols="20")

	return ws


def insert_to_gsheet(
		sheet_name, 
		file_key, 
		row = None,
		rows = None,
		create_sheet=False,
		overwrite_existing_sheet=False, 
		return_sheet=False,
		clear_sheet=False):

	assert row or rows, "Provide `row` or `rows`"
	gc = gs.service_account(filename='/tmp/creds.json')
	ws = get_worksheet(gc, create_sheet, file_key, sheet_name, overwrite_existing_sheet)
	if clear_sheet:
		ws.clear()

	row_index = [i for (i,j) in enumerate(ws.col_values(1)) if j == '']
	if row_index:
		row_index = row_index[0]
	else:
		row_index = len([value for value in ws.col_values(1) if value])
	if row:
		ws.insert_rows([row], row_index + 1)
	elif rows:
		ws.insert_rows(rows, row_index + 1)

	if return_sheet:
		return ws