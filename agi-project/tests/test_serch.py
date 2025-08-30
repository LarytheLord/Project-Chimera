from googlesearch import search
from bs4 import BeautifulSoup
import requests
# The search query
query = "python programming examples"

# Perform the search and get the first 5 results
try:
    results = search(query, num_results=5) 
    u=[]
    for i, url in enumerate(results):
        print(f"{url}")
        u.append(url)
    print(u[1])
    e=requests.get(u[1])
    t=e.text

    b=BeautifulSoup(t,"html.parser")

    print(b.get_text())


except Exception as e:
    print(f"An error occurred: {e}")

