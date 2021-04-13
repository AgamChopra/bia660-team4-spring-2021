# WEB MINING PROJECT (DATA COLLECTION)
# Group 4: Arnold Yanga, Agamdeep Chopre, Ilesh Sharda
#
#   HIGHLIGHTS:
#       - The following code autonomously scrapes coincheckup ICO profiles
#       - It successfully searches each ICO page and checks for Whitepaper pdf download links
#       - The code then downloads the each pdf onto the working directory
#       - 11 out of the 38 available PDFs failed to upload
#       - The algorithm should work for other categories (i.e. Archived, Current Pre-sale, etc.)
#       - Further testing is needed
#   TO DO:
#       - Troubleshoot why some pdf downloads fail to be imported
#       - Organize functions into a class in order to simplify the script
#       - Clean the data
#       - Explore further variables to scrape from coincheck

##########################  START OF FUNCTION DEFINITIONS  ###############################
import os
import requests
import PyPDF2 as p2
import pandas as pd
from pathlib import Path 
from bs4 import BeautifulSoup

def scrape(website):
    '''
    Scrapes coincheckup for ICO profile extensions
    
    INPUT:
    website :
        Link for the coincheckup ICO page (can very between Current, Archived, etc.)

    OUTPUT:
    list
        List of ICO profile url extensions 

    '''
    page = requests.get(website)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.select('a.ghost-link')
    return [a['href'] for a in divs]

def getICOname(idx, extension_list):
    '''
    Forms a list that stores the names of each ICO containing a PDF link

    '''
    names = [name.strip().replace("/", "").replace("-", " ") for name in extension_list]
    return [names[i] for i in idx]

def getPDFlink(extension_list):
    """
    
    Scrapes each ICO profile page and collects the Whitepaper download links if available.
    Indexes are recorded to keep track of ICO's without download links'

    """
    website = "http://ico.coincheckup.com"
    indexes = []
    link_list = []
    links = extension_list
    
    for idx, link in enumerate(links):
        profileURL = website + link
        page = requests.get(profileURL)

        if page.status_code == 200:
            soup = BeautifulSoup(page.content, 'html.parser')
            a = soup.select("a.button")
            if len(a) > 3:
                pdfLink = a[3]['href']
                link_list.append(pdfLink)
                indexes.append(idx)
    return indexes, link_list

def extractData(file):
    '''
    
    PARAMETERS:
    file : TYPE
        A PDF file containing the ICO whitepaper or equivalent file

    OUTPUT
    TEXT
        Text data (not clean) and some may contain the filler message ("Failed to import")
    '''
    
    PDFfile = open(file, "rb")
    
    pdfread = p2.PdfFileReader(PDFfile)
    pageNum = pdfread.numPages
    text = ''
    
    for idx in range(pageNum):
        pageObj = pdfread.getPage(idx)
        text += pageObj.extractText()

    PDFfile.close()
    return str(text)
   
##########################  END OF FUNCTION DEFINITIONS  ###############################


##########################  BEGIN IMPLEMENTATION  ###############################

# Define coincheckup url and user directory
website = "http://ico.coincheckup.com"
directory = "/Users/ArnoldYanga/Web Mining/Project"


URL_extensions = scrape(website)                    # Get the url extensions for all ICO's on the page
idx, list_of_links = getPDFlink(URL_extensions)     # Get the links of all available whitepapers
names = getICOname(idx, URL_extensions)             # Extract the names of all ICOs collected
results = zip(names, list_of_links)                 # Zip the names and links together and store in DF

df = pd.DataFrame(set(results), columns = ['ICO', 'Whitepaper Link'])
 
##### Download the available PDFs ######

for i in range(len(df)):
    newfile = df['ICO'][i].replace(" ", "_")+".pdf"     # Create new ICO file name
    filename = Path(newfile)                            # Establish file path
    url = df['Whitepaper Link'][i]                      # Select whitepaper links
    response = requests.get(url)                        # REquest and download pdf files
    filename.write_bytes(response.content)

##### Extract the White Paper text data and store into a dataframe ######

ICOtext = []

# Iterate through all pdf's in the user's directory 
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".pdf"): 
        try:
            text = extractData(filename)
            ICOtext.append(text)
        except:
            ICOtext.append("Failed to import")
    else:
        continue

df = df.sort_values('ICO')
df['text'] = ICOtext
df.head(10)

##########################  END OF IMPLEMENTATION  ###############################
