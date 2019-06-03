import sqlite3
import re
con = sqlite3.connect('C:/Windows.old/Users/acces/Local Settings/Google/Chrome/User Data/Default/History')
cursor = con.cursor()
cursor.execute("SELECT url FROM urls")
urls = cursor.fetchall()
for url in urls:
    
    u=str(url)
    length=len(u)
    url1=u[2:length-3]
    
    aa=re.findall('[b]+[u]+[g]+',url1)
    """a=str(aa)
    length1=len(a)
    a1=a[2:length1-2]"""
    if(aa==['bug']):
        print('url is blocked')
    else:
        print(url1)
