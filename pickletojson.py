# convert_pickle_to_json.py
import pickle, json
from google.oauth2.credentials import Credentials

with open('token.pickle','rb') as f:
    creds = pickle.load(f)   # creds is a google.oauth2.credentials.Credentials object

with open('token.json','w') as f:
    f.write(creds.to_json())

print('token.json written')