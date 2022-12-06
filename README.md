# group-project-the-fast-the-curious
group-project-the-fast-the-curious created by GitHub Classroom

# Group Members
Georgi Nikolov (gnikol5@uic.edu)&emsp;(Github: https://github.com/NikolovG)<br>
Daniel Valencia (dvalen2@uic.edu)&emsp;(Github: https://github.com/Valencia24)<br>
Lizbeth Gutierrez (igutie37@uic.edu)&emsp; (Github: https://github.com/lgutie37)<br>
Jovad Uribe (juribe5@uic.edu)&emsp; (Github: https://github.com/jovuribe)<br>

# Python Modules (Not Native)
Numpy, Pandas, Meteostat, Prophet, DateTime, Sklearn

### Original Data Reference:
[Rapid API](https://rapidapi.com/meteostat/api/meteostat)

### Original Data Retrieval:
import pandas as pd  
import requests  
  
url = "https://meteostat.p.rapidapi.com/stations/daily"  
querystring = {"station":"KPWK0","start":"2012-01-01","end":"2021-01-01","model":"true","units":"imperial"}  
headers = {  
	"X-RapidAPI-Key": "6fec950325msha54f5bc890b65e0p1fd8d0jsnc882976e1304",  
	"X-RapidAPI-Host": "meteostat.p.rapidapi.com"  
}  
response = requests.request("GET", url, headers=headers, params=querystring)  
json = response.json()  
type(json['data'])  
type(json['data'][0])  
df = pd.DataFrame(json['data'])  

### Cleaning Scripts:
df = df.drop("snow", axis='columns')  
df = df.drop("wdir", axis='columns')  
df = df.drop("wspd", axis='columns')  
df = df.drop("wpgt", axis='columns')  
df = df.drop("tsun", axis='columns')  
  
df[df['prcp'].isna()]  
df[df['pres'].isna()]  
  
df = df.dropna(subset=['prcp','pres'])  
  
df['date'] = df['date'].replace('-', '', regex=True).astype(int)  
df = df.dropna(subset=['tavg','tmin','tmax'])  
