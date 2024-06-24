

!pip install --quiet openml
import openml
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

#this will display a list of available datasets for download
availDatasets = openml.datasets.list_datasets(output_format='dataframe')
availDatasets

#download the selected dataset using the did column:
#use the did column
queriesDS = fetch_openml(data_id=41701, as_frame=True, parser='auto')
print(queriesDS.DESCR)
print(queriesDS.feature_names)
print(queriesDS.target_names)

from turbodbc import connect, make_options
options = make_options(autocommit=True)

#connect to impala or hadoop in this case
connection = connect(dsn="Hadoop", turbodbc_options=options)
connection2 = connect(dsn='Impala_Prod', turbodbc_options=options)
cursor = connection2.cursor()
query = """
select * ....
"""
cursor.execute(query)
df = cursor.fetchall()
df = pd.DataFrame(df)

df.rename(columns={
    0: 'col1', 1: 'col2', 2: 'col3', 3: 'col4'
    }, inplace=True)

import urllib
from sqlalchemy import create_engine
driver = "{ODBC Driver 17 for SQL Server}"
server = "Server"
database = "Database"
user = "username"
password = "pasw"
conn = f"""Driver={driver};Server=tcp:{server},1433;Database={database};Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""
params = urllib.parse.quote_plus(conn)
conn_str = 'mssql+pyodbc:///?autocommit=true&odbc_connect={}'.format(params)
sql_engine = sqlalchemy.create_engine(conn_str, echo=True, fast_executemany = True)
conn = sql_engine.connect()

query = """
  select * from ....
 """
df = pd.read_sql(query, conn, coerce_float=False)

#copy to SQL (pandas)
driver = "{ODBC Driver 17 for SQL Server}"
server = "server"
database = "db"
user = "username"
password = "psw"
conn = f"""Driver={driver};Server=tcp:{server},1433;Database={database};Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""
params = urllib.parse.quote_plus(conn)
conn_str = 'mssql+pyodbc:///?autocommit=False&odbc_connect={}'.format(params) #autocommit false LOCKS the table! it is faster but more complicated if the table has many accesses
sql_engine = sqlalchemy.create_engine(conn_str, echo=True, fast_executemany = True, pool_pre_ping=True, pool_recycle=3600)
conn = sql_engine.connect()
df.to_sql(name = "table",if_exists="append",index=False, schema='dbo',chunksize=100000,con=conn)


import io
import os
from azure.storage.blob import BlobServiceClient, ContainerClient
import zipfile
import time
import pandas as pd
import numpy
from concurrent.futures import ThreadPoolExecutor, as_completed

sourceconstring = "BlobEndpoint=https://xxxxxx.blob.core.windows.net/;QueueEndpoint=https://xxxxx.queue.core.windows.net/;FileEndpoint=https://xxxxx.file.core.windows.net/;TableEndpoint=https://xxxxx.table.core.windows.net/;SharedAccessSignature=sv=kkkkkkkkkkkkkkkkk"

# Create a BlobClient object with SAS authorization
sourceblob_client_sas = BlobServiceClient.from_connection_string(sourceconstring)
sourcecontainers = sourceblob_client_sas.list_containers()

#identify a container
tgtcontainer = sourceblob_client_sas.get_container_client('xxxxx')

#delete and recreate the container (easier way to clean it)
tgtcontainer.delete_container()
retry = 1
while(retry>0):
 try:
  tgtcontainer = sourceblob_client_sas.create_container('xxxxx')
  retry = -1
 except Exception as e:
  err = str(e)
  if "The specified container is being deleted" in err:
   retry = retry + 1
  else:
   retry = 0

#download everything using multi-threading
scontainers = []
for cont in sourcecontainers:
 scontainers.append(cont)
scontainers.sort(key=lambda x: x.name, reverse=False)
chunks = [scontainers[x:x+50] for x in range(0, len(scontainers), 50)]

#function: what is being done with the files in the containers
def processing(cc):
  currentcontainer = cc.name
  print("doing container: " + currentcontainer )
  try:
   sourcecontainer = sourceblob_client_sas.get_container_client(cc.name)
   sourcefiles = sourcecontainer.list_blobs()
   for ff in sourcefiles:
    filename = ff.name
    filetime = ff.last_modified
    if ((".zip" in filename)):
     source_blob = sourcecontainer.get_blob_client(ff)
     #destination to copy the file (if needed)
     with open(file=os.path.join(r'E://folder/', "filename.zip"), mode="wb") as sample_blob:
      download_stream = source_blob.download_blob()
      sample_blob.write(download_stream.readall())
      try:
       with zipfile.ZipFile('E://folder/filename.zip', 'r') as zipObj:
         for entry in zipObj.infolist():
          #entry is each file in the zip
          if ((".zip" in filename)):
           if ("codition of file name" in entry.filename):
            zipObj.extract(entry, 'E://extractLocation')
            with open(file=os.path.join('E://extractLocation//', entry.filename), mode="r") as data:
             file_lines = []
             for line in data.readlines():
              try:
               #do something per line
              except Exception as e:
               print(e)
            with open(file=os.path.join('E://extractLocation//', entry.filename), mode="w") as data:
             data.writelines(file_lines)
            with open(file=os.path.join('E://extractLocation//', entry.filename), mode="rb") as data:
             #upload to Azure
             blob_client = tgtcontainer.upload_blob(name=entry.filename, data=data, overwrite=True)
            #delete local copy
            os.remove('E://extractlocation//' + entry.filename)
      except Exception as e:
       print(e + "," + num + "," + filename)
     os.remove('E://folder//filename.zip')
  except Exception as e:
   print(e + "," + currentcontainer)

processed = []
curcounterst = 0

#process in the multi-thread
for chunk in chunks:
 curcounterst = curcounterst + 1
 chunk.sort(key=lambda x: x.name, reverse=False)
 firstcontainer = chunk[0].name
 lastcontainer = chunk[len(chunk)-1].name
 print("start of chunk " + str(curcounterst) + " of " + str(len(chunks)) + " from " + firstcontainer + " to " + lastcontainer)
 errors = []
 usage = []
 longer = []
 accesses = []
 with ThreadPoolExecutor() as executor:
    # submit tasks and collect futures
    futures = executor.map(processing, chunk)
    # process task results as they are available
    futured = as_completed(futures)


from azure.cosmos import exceptions, CosmosClient, PartitionKey
from azure.cosmos import cosmos_client
import pandas as pd
import numpy as np
import json

url = "https://location:443/"
key = "seckey"
client = cosmos_client.CosmosClient(url, key)

querytext = 'SELECT * from ....'

database = client.get_database_client("database")
container = database.get_container_client("container")

items = container.query_items(query=querytext,enable_cross_partition_query=True)
a = []
for item in items:
 dfshort = pd.DataFrame.from_dict(json.loads(json.dumps(item, indent=True)))
 a.append(dfshort)
dataf = pd.concat(a, axis=0, ignore_index=True)
#json is contained in Data in this case
df2 = pd.json_normalize(dataf['Data'])
dataf = pd.concat([dataf, df2], axis=1)




